/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {BackendTimingInfo, DataMover, KernelBackend} from './kernels/backend';
import {Profiler} from './profiler';
import {backpropagateGradients, getFilteredNodesXToY, NamedGradientMap, TapeNode} from './tape';
import {DataId, Tensor, Tensor3D, TensorTracker, Variable} from './tensor';
import {NamedTensorMap, NamedVariableMap, TensorContainer} from './tensor_types';
import {getTensorsInContainer, isTensorInList} from './tensor_util';
import {DataType, DataValues} from './types';
import * as util from './util';
import {bytesFromStringArray, makeOnesTypedArray, now, sizeFromShape} from './util';

/**
 * A function that computes an output. The save function is for saving tensors
 * computed in the forward pass, that we need in the backward pass.
 */
export type ForwardFunc<T> =
    (backend: KernelBackend, save?: <S extends Tensor>(tensor: S) => S) => T;

/**
 * @docalias (a: Tensor, b: Tensor,...) => {
 * value: Tensor, * gradFunc: (dy: Tensor) => Tensor | Tensor[] * }
 */
export type CustomGradientFunc<T extends Tensor> = (...args: Tensor[]) => {
  value: T, gradFunc: (dy: T) => Tensor | Tensor[];
};

export type MemoryInfo = {
  numTensors: number; numDataBuffers: number; numBytes: number;
  unreliable?: boolean; reasons: string[];
};

type KernelProfile = {
  name: string; bytesAdded: number; totalBytesSnapshot: number;
  tensorsAdded: number;
  totalTensorsSnapshot: number;
  inputShapes: number[][];
  outputShape: number[] | number[][];
};

export type ProfileInfo = {
  newBytes: number; newTensors: number; peakBytes: number;
  kernels: KernelProfile[];
  result: TensorContainer;
};

export interface TimingInfo extends BackendTimingInfo {
  wallMs: number;
}

/** @docalias Function */
export type ScopeFn<T extends TensorContainer> = () => T;

export interface TensorManager {
  registerTensor(a: Tensor, backend?: KernelBackend): void;
  registerVariable(v: Variable): void;
  disposeTensor(a: Tensor): void;
  memory(): {numDataBuffers: number; numBytes: number;};
}

interface ScopeState {
  track: Tensor[];
  name: string;
}

export class Engine implements TensorManager, TensorTracker, DataMover {
  // Public since optimizers will use it.
  registeredVariables: NamedVariableMap = {};

  private nextTapeNodeId = 0;
  private numBytes = 0;
  private numTensors = 0;
  private numStringTensors = 0;
  private numDataBuffers = 0;

  private profiling = false;
  private activeProfile: ProfileInfo;

  private activeTape: TapeNode[];
  private gradientScopeCount = 0;
  private customGradientDepth = 0;

  // Keep Tensors that parallel the tapes.
  private activeScope: ScopeState;
  private scopeStack: ScopeState[] = [];
  private keepTensors: Set<number> = new Set();
  private profiler: Profiler;

  private tensorInfo = new WeakMap<DataId, {
    backend: KernelBackend,
    bytes: number,
    dtype: DataType,
    shape: number[],
    refCount: number
  }>();

  constructor(
      public backend: KernelBackend, public safeMode: boolean,
      private debugMode: () => boolean) {
    this.profiler = new Profiler(backend);
    this.activeProfile =
        {newBytes: 0, newTensors: 0, peakBytes: 0, kernels: [], result: null};
  }

  moveData(dataId: DataId) {
    this.write(dataId, this.readSync(dataId));
  }

  tidy<T extends TensorContainer>(
      nameOrFn: string|ScopeFn<T>, fn?: ScopeFn<T>, gradMode = false): T {
    // gradMode Primarily for internal use during backprop
    //          If true, will start a tape if it is the outermost tidy.

    let name: string = null;
    if (fn == null) {
      // Called with only 1 argument.
      if (typeof nameOrFn !== 'function') {
        throw new Error('Please provide a function to tidy()');
      }
      fn = nameOrFn;
    } else {
      // Called with 2 arguments.
      if (typeof nameOrFn !== 'string' && !(nameOrFn instanceof String)) {
        throw new Error(
            'When calling with two arguments, the first argument ' +
            'to tidy() must be a string');
      }
      if (typeof fn !== 'function') {
        throw new Error(
            'When calling with two arguments, the 2nd argument ' +
            'to tidy() must be a function');
      }
      name = nameOrFn as string;
      // TODO(nsthorat,smilkov): Do operation logging and performance
      // profiling.
    }
    let result: T;
    return this.scopedRun(
        () => this.startScope(name, gradMode),
        () => this.endScope(result, gradMode), () => {
          result = fn();
          if (result instanceof Promise) {
            console.error('Cannot return a Promise inside of tidy.');
          }
          return result;
        });
  }

  private scopedRun<T>(start: () => void, end: () => void, f: () => T): T {
    start();
    try {
      const res = f();
      end();
      return res;
    } catch (ex) {
      end();
      throw ex;
    }
  }

  private static nextTensorId = 0;
  nextTensorId(): number {
    return Engine.nextTensorId++;
  }

  private static nextVariableId = 0;
  nextVariableId(): number {
    return Engine.nextVariableId++;
  }

  runKernel<T extends Tensor|Tensor[], I extends NamedTensorMap>(
      forwardFunc: ForwardFunc<T>,
      inputs: I,
      backwardsFunc?: (dy: T, saved: Tensor[]) => {[P in keyof I]: () => I[P]},
      ): T {
    let result: T;
    const saved: Tensor[] = [];
    const saveFunc = <T extends Tensor>(x: T): T => {
      saved.push(x);
      return x;
    };
    const scopeName = this.activeScope != null ? this.activeScope.name : '';
    const startingBytecount = this.numBytes;
    const startingNumTensors = this.numTensors;

    // Stop recording to a tape when running a kernel.
    this.scopedRun(
        () => this.customGradientDepth++, () => this.customGradientDepth--,
        () => {
          if (!this.debugMode()) {
            result = forwardFunc(this.backend, saveFunc);
          } else {
            result = this.profiler.profileKernel(
                scopeName, () => forwardFunc(this.backend, saveFunc));
          }
        });

    if (this.shouldRecord()) {
      const tapeNode: TapeNode = {
        id: this.nextTapeNodeId++,
        name: scopeName,
        inputs,
        outputs: Array.isArray(result) ? result : [result] as Tensor[]
      };
      if (backwardsFunc != null) {
        tapeNode.gradient =
            ((dy: T) => backwardsFunc(dy, saved)) as (dy: Tensor) =>
                NamedGradientMap;
      }
      this.activeTape.push(tapeNode);
    }

    if (this.profiling) {
      this.activeProfile.kernels.push({
        name: scopeName,
        bytesAdded: this.numBytes - startingBytecount,
        totalBytesSnapshot: this.numBytes,
        tensorsAdded: this.numTensors - startingNumTensors,
        totalTensorsSnapshot: this.numTensors,
        inputShapes: Object.keys(inputs).map(key => inputs[key].shape),
        outputShape: Array.isArray(result) ?
            (result as Tensor[]).map(item => (item as Tensor).shape) :
            (result as Tensor).shape
      });
    }

    return result;
  }

  // TensorManager implementation.

  registerTensor(a: Tensor|Variable, backend?: KernelBackend): void {
    const refCount = this.tensorInfo.has(a.dataId) ?
        this.tensorInfo.get(a.dataId).refCount :
        0;
    this.numTensors++;
    if (a.dtype === 'string') {
      this.numStringTensors++;
    }
    if (refCount === 0) {
      this.numDataBuffers++;

      // Bytes for complex numbers are counted by their components. Bytes for
      // string tensors are counted when writing values.
      let bytes = 0;
      if (a.dtype !== 'complex64' && a.dtype !== 'string') {
        bytes = util.sizeFromShape(a.shape) * util.bytesPerElement(a.dtype);
      }
      this.tensorInfo.set(a.dataId, {
        backend: backend != null ? backend : this.backend,
        dtype: a.dtype,
        shape: a.shape,
        bytes,
        refCount: 0
      });
      this.numBytes += bytes;
      if (backend != null) {
        backend.register(a.dataId, a.shape, a.dtype);
      } else {
        this.backend.register(a.dataId, a.shape, a.dtype);
      }
    }
    this.tensorInfo.get(a.dataId).refCount++;
    if (!(a instanceof Variable)) {
      this.track(a);
    }
  }

  registerVariable(v: Variable) {
    if (this.registeredVariables[v.name] != null) {
      throw new Error(`Variable with name ${v.name} was already registered`);
    }
    this.registeredVariables[v.name] = v;
  }

  disposeTensor(a: Tensor): void {
    if (!this.tensorInfo.has(a.dataId)) {
      return;
    }
    if (this.keepTensors.has(a.id)) {
      this.keepTensors.delete(a.id);
    }
    this.numTensors--;
    if (a.dtype === 'string') {
      this.numStringTensors--;
    }
    const info = this.tensorInfo.get(a.dataId);
    const refCount = info.refCount;
    if (refCount <= 1) {
      // Don't count bytes for complex numbers as they are counted by their
      // components.
      if (a.dtype !== 'complex64') {
        this.numBytes -= info.bytes;
      }
      this.numDataBuffers--;
      info.backend.disposeData(a.dataId);
      this.tensorInfo.delete(a.dataId);
    } else {
      this.tensorInfo.get(a.dataId).refCount--;
    }
    // TODO(nsthorat): Construct an error and save the stack trace for
    // debugging when in debug mode. Creating a stack trace is too expensive
    // to do unconditionally.
  }

  disposeVariables(): void {
    for (const varName in this.registeredVariables) {
      const v = this.registeredVariables[varName];
      this.disposeTensor(v);
      delete this.registeredVariables[varName];
    }
  }

  memory(): MemoryInfo {
    const info = this.backend.memory() as MemoryInfo;
    info.numTensors = this.numTensors;
    info.numDataBuffers = this.numDataBuffers;
    info.numBytes = this.numBytes;
    if (this.numStringTensors > 0) {
      info.unreliable = true;
      if (info.reasons == null) {
        info.reasons = [];
      }
      info.reasons.push(
          'Memory usage by string tensors is approximate ' +
          '(2 bytes per character)');
    }
    return info;
  }

  async profile(query: () => TensorContainer): Promise<ProfileInfo> {
    this.profiling = true;

    const startBytes = this.numBytes;
    const startNumTensors = this.numTensors;

    this.activeProfile.kernels = [];
    this.activeProfile.result = query();

    this.profiling = false;

    this.activeProfile.peakBytes =
        Math.max(...this.activeProfile.kernels.map(d => d.totalBytesSnapshot));
    this.activeProfile.newBytes = this.numBytes - startBytes;
    this.activeProfile.newTensors = this.numTensors - startNumTensors;
    return this.activeProfile;
  }

  private shouldRecord(): boolean {
    return this.activeTape != null && this.customGradientDepth === 0;
  }

  private addTapeNode(
      inputs: Tensor[], result: Tensor,
      gradientsFunc: (dy: Tensor) => Tensor[]): void {
    const inputsMap: NamedTensorMap = {};
    inputs.forEach((input, idx) => {
      inputsMap[idx] = input;
    });

    const gradient = (dy: Tensor) => {
      const res = gradientsFunc(dy);
      const resMap: NamedGradientMap = {};
      res.forEach((r, idx) => {
        resMap[idx] = () => r;
      });
      return resMap;
    };

    const tapeNode: TapeNode = {
      id: this.nextTapeNodeId++,
      name: this.activeScope.name,
      inputs: inputsMap,
      outputs: [result],
      gradient
    };
    this.activeTape.push(tapeNode);
  }

  keep<T extends Tensor>(result: T): T {
    if (this.scopeStack.length === 1 && this.safeMode) {
      throw new Error(
          'Safe mode is ON. Enclose all tensor operations inside tf.tidy(): ' +
          'tf.tidy(() => {...}) to avoid memory leaks.');
    }
    this.keepTensors.add(result.id);
    return result;
  }

  /**
   * Start a scope. Use this with endScope() to achieve the same functionality
   * as scope() without the need for a function closure.
   */
  startScope(name?: string, gradientsMode = false) {
    if (gradientsMode && this.gradientScopeCount === 0) {
      this.activeTape = [];
    }
    if (gradientsMode) {
      this.gradientScopeCount++;
    }

    const scopeInfo: ScopeState = {track: [], name: 'unnamed scope'};
    if (name) {
      scopeInfo.name = name;
    }
    this.scopeStack.push(scopeInfo);
    this.activeScope = scopeInfo;
  }

  /**
   * End a scope. Use this with startScope() to achieve the same functionality
   * as scope() without the need for a function closure.
   */
  endScope(result?: TensorContainer, gradientsMode = false) {
    if (gradientsMode) {
      this.gradientScopeCount--;
      if (this.gradientScopeCount === 0) {
        this.activeTape = null;
      }
    }

    const tensorsToKeep = new Set(this.keepTensors);

    const tensorsToTrackInParent = getTensorsInContainer(result);
    tensorsToTrackInParent.forEach(tensor => tensorsToKeep.add(tensor.id));

    // Dispose the arrays tracked in this scope.
    for (let i = 0; i < this.activeScope.track.length; i++) {
      const tensor = this.activeScope.track[i];
      if (tensorsToKeep.has(tensor.id)) {
        continue;
      }

      if (this.activeTape != null) {
        tensorsToTrackInParent.push(tensor);
      } else {
        tensor.dispose();
      }
    }

    const oldScope = this.scopeStack.pop();
    this.activeScope = this.scopeStack.length === 0 ?
        null :
        this.scopeStack[this.scopeStack.length - 1];

    // Track the current result in the parent scope.
    tensorsToTrackInParent.forEach(tensor => {
      // Only track the tensor if was allocated in the inner scope and is not
      // globally kept.
      if (!this.keepTensors.has(tensor.id) &&
          isTensorInList(tensor, oldScope.track)) {
        this.track(tensor);
      }
    });
  }

  /**
   * Returns gradients of `f` with respect to each of the `xs`. The gradients
   * returned are of the same length as `xs`, but some might be null if `f` was
   * not a function of that `x`. It also takes optional dy to multiply the
   * gradient, which defaults to `1`.
   */
  gradients<T extends Tensor>(
      f: () => T, xs: Tensor[], dy?: T,
      allowNoGradients = false): {value: T, grads: Tensor[]} {
    util.assert(
        xs.length > 0, () => 'gradients() received an empty list of xs.');
    if (dy != null && dy.dtype !== 'float32') {
      throw new Error(`dy must have 'float32' dtype, but has '${dy.dtype}'`);
    }

    return this.tidy('gradients', () => {
      const y = f();
      util.assert(
          y instanceof Tensor,
          () => 'The result y returned by f() must be a tensor.');
      // Filter out the nodes that don't connect x => y.
      const filteredTape = getFilteredNodesXToY(this.activeTape, xs, y);
      if (!allowNoGradients && filteredTape.length === 0 && xs.length > 0) {
        throw new Error(
            'Cannot compute gradient of y=f(x) with respect to x. Make sure ' +
            'that the f you passed encloses all operations that lead from x ' +
            'to y.');
      }

      const accumulatedGradientMap: {[tensorId: number]: Tensor} = {};
      accumulatedGradientMap[y.id] = (dy == null) ? ones(y.shape) : dy;

      // Backprop gradients through the filtered nodes.
      backpropagateGradients(accumulatedGradientMap, filteredTape);

      const grads = xs.map(x => accumulatedGradientMap[x.id]);
      return {value: y, grads};
    }, true /* gradientsMode */);
  }

  customGrad<T extends Tensor>(f: CustomGradientFunc<T>):
      (...args: Tensor[]) => T {
    util.assert(
        util.isFunction(f),
        () => 'The f passed in customGrad(f) must be a function.');
    return (...inputs: Tensor[]): T => {
      util.assert(
          inputs.every(t => t instanceof Tensor),
          () => 'The args passed in customGrad(f)(x1, x2,...) must all be ' +
              'tensors');

      let gradientsFunc: (dy: T) => Tensor | Tensor[];
      let result: T;
      this.scopedRun(
          () => this.customGradientDepth++, () => this.customGradientDepth--,
          () => {
            const gradientsMode = true;
            result = this.tidy(f.name, () => {
              const {value, gradFunc} = f(...inputs);
              util.assert(
                  value instanceof Tensor,
                  () =>
                      'The function f passed in customGrad(f) must return an ' +
                      'object where `obj.value` is a tensor');
              util.assert(
                  util.isFunction(gradFunc),
                  () =>
                      'The function f passed in customGrad(f) must return an ' +
                      'object where `obj.gradFunc` is a function.');
              gradientsFunc = gradFunc;
              return value;
            }, gradientsMode);
          });

      if (this.shouldRecord()) {
        const gradFunc = (dy: T): Tensor[] => {
          const res = gradientsFunc(dy);
          const grads: Tensor[] = Array.isArray(res) ? res : [res];
          util.assert(
              grads.length === inputs.length,
              () => 'The function f passed in customGrad(f) must return an ' +
                  'object where `obj.gradFunc` is a function that returns ' +
                  'the same number of tensors as inputs passed to f(...).');
          util.assert(
              grads.every(t => t instanceof Tensor),
              () => 'The function f passed in customGrad(f) must return an ' +
                  'object where `obj.gradFunc` is a function that returns a ' +
                  'list of only tensors.');
          return grads;
        };
        this.addTapeNode(inputs, result, gradFunc);
      }
      return result;
    };
  }

  // Forwarding to backend.
  write(dataId: DataId, values: DataValues): void {
    const info = this.tensorInfo.get(dataId);
    // Bytes for string tensors are counted when writing.
    if (info.dtype === 'string') {
      const newBytes = bytesFromStringArray(values as string[]);
      this.numBytes += newBytes - info.bytes;
      info.bytes = newBytes;
    }

    if (this.backend !== info.backend) {
      // Delete the tensor from the old backend and move it to the new backend.
      info.backend.disposeData(dataId);
      info.backend = this.backend;
      this.backend.register(dataId, info.shape, info.dtype);
    }
    this.backend.write(dataId, values);
  }
  readSync(dataId: DataId): DataValues {
    // Route the read to the correct backend.
    const info = this.tensorInfo.get(dataId);
    return info.backend.readSync(dataId);
  }
  read(dataId: DataId): Promise<DataValues> {
    // Route the read to the correct backend.
    const info = this.tensorInfo.get(dataId);
    return info.backend.read(dataId);
  }
  fromPixels(
      pixels: ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement,
      numChannels: number): Tensor3D {
    return this.backend.fromPixels(pixels, numChannels);
  }
  async time(query: () => void): Promise<TimingInfo> {
    const start = now();
    const timingInfo = await this.backend.time(query) as TimingInfo;
    timingInfo.wallMs = now() - start;
    return timingInfo;
  }

  /**
   * Tracks a Tensor in the current scope to be automatically cleaned up
   * when the current scope ends, and returns the value.
   *
   * @param result The Tensor to track in the current scope.
   */
  private track<T extends Tensor>(result: T): T {
    if (this.scopeStack.length === 1 && this.safeMode) {
      throw new Error(
          'Safe mode is ON. Enclose all tensor operations inside tf.tidy(): ' +
          'tf.tidy(() => {op();...}); to avoid memory leaks.');
    }
    if (this.activeScope != null) {
      this.activeScope.track.push(result);
    }
    return result;
  }
}

function ones(shape: number[]): Tensor {
  const values = makeOnesTypedArray(sizeFromShape(shape), 'float32');
  return Tensor.make(shape, {values});
}
