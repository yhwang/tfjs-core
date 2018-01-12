/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
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

// tslint:disable:restrict-plus-operands
// tslint:disable-next-line:max-line-length
import {AdagradOptimizer, Array2D, CostReduction, FeedEntry, Graph, InGPUMemoryShuffledInputProviderBuilder, NDArray, NDArrayMath, Session, Tensor} from 'deeplearn';
import {Array1D} from 'deeplearn/dist/math/ndarray';

/** Generates GameOfLife sequence pairs (current sequence + next sequence) */
export class GameOfLife {
  math: NDArrayMath;
  size: number;

  constructor(size: number, math: NDArrayMath) {
    this.math = math;
    this.size = size;
  }

  setSize(size: number) {
    this.size = size;
  }

  generateGolExample(): [NDArray, NDArray] {
    let world: NDArray;
    let worldNext: NDArray;
    this.math.scope(keep => {
      const randWorld =
          Array2D.randUniform([this.size - 2, this.size - 2], 0, 2, 'int32');
      const worldPadded = this.math.pad2D(randWorld, [[1, 1], [1, 1]]);
      const numNeighbors = this.countNeighbors(this.size, worldPadded);

      const cellRebirths =
          this.math.equal(numNeighbors, Array1D.new([3], 'int32'));

      const cellSurvives = this.math.logicalOr(
          cellRebirths,
          this.math.equal(numNeighbors, Array1D.new([2], 'int32')));

      const survivors = this.math.where(
          cellSurvives, randWorld, Array2D.zerosLike(randWorld));

      const nextWorld =
          this.math.where(cellRebirths, Array2D.onesLike(randWorld), survivors);

      world = keep(worldPadded);
      worldNext = keep(this.math.pad2D(nextWorld as Array2D, [[1, 1], [1, 1]]));
    });
    return [world, worldNext];
  }

  /** Counts total sum of neighbors for a given world. */
  private countNeighbors(size: number, worldPadded: Array2D): Array2D {
    let neighborCount = this.math.add(
        this.math.slice2D(worldPadded, [0, 0], [size - 2, size - 2]),
        this.math.slice2D(worldPadded, [0, 1], [size - 2, size - 2]));
    neighborCount = this.math.add(
        neighborCount,
        this.math.slice2D(worldPadded, [0, 2], [size - 2, size - 2]));
    neighborCount = this.math.add(
        neighborCount,
        this.math.slice2D(worldPadded, [1, 0], [size - 2, size - 2]));
    neighborCount = this.math.add(
        neighborCount,
        this.math.slice2D(worldPadded, [1, 2], [size - 2, size - 2]));
    neighborCount = this.math.add(
        neighborCount,
        this.math.slice2D(worldPadded, [2, 0], [size - 2, size - 2]));
    neighborCount = this.math.add(
        neighborCount,
        this.math.slice2D(worldPadded, [2, 1], [size - 2, size - 2]));
    neighborCount = this.math.add(
        neighborCount,
        this.math.slice2D(worldPadded, [2, 2], [size - 2, size - 2]));
    return neighborCount as Array2D;
  }
}

/**
 * Main class for running a deep-neural network of training for Game-of-life
 * next sequence.
 */
export class GameOfLifeModel {
  session: Session;
  math: NDArrayMath;

  optimizer: AdagradOptimizer;
  inputTensor: Tensor;
  targetTensor: Tensor;
  costTensor: Tensor;
  predictionTensor: Tensor;

  size: number;
  batchSize: number;
  step = 0;

  // Maps tensors to InputProviders
  feedEntries: FeedEntry[];

  constructor(math: NDArrayMath) {
    this.math = math;
  }

  setupSession(
      boardSize: number, batchSize: number, initialLearningRate: number,
      numLayers: number, useLogCost: boolean): void {
    this.optimizer = new AdagradOptimizer(initialLearningRate);

    this.size = boardSize;
    this.batchSize = batchSize;
    const graph = new Graph();
    const shape = this.size * this.size;

    this.inputTensor = graph.placeholder('input', [shape]);
    this.targetTensor = graph.placeholder('target', [shape]);

    let hiddenLayer = GameOfLifeModel.createFullyConnectedLayer(
        graph, this.inputTensor, 0, shape);
    for (let i = 1; i < numLayers; i++) {
      // Last layer will use a sigmoid:
      hiddenLayer = GameOfLifeModel.createFullyConnectedLayer(
          graph, hiddenLayer, i, shape, i < numLayers - 1);
    }

    this.predictionTensor = hiddenLayer;

    if (useLogCost) {
      this.costTensor =
          this.logLoss(graph, this.targetTensor, this.predictionTensor);
    } else {
      this.costTensor =
          graph.meanSquaredCost(this.targetTensor, this.predictionTensor);
    }
    this.session = new Session(graph, this.math);
  }

  trainBatch(fetchCost: boolean, worlds: Array<[NDArray, NDArray]>): number {
    this.setTrainingData(worlds);

    let costValue = -1;
    this.math.scope(() => {
      const cost = this.session.train(
          this.costTensor, this.feedEntries, this.batchSize, this.optimizer,
          fetchCost ? CostReduction.MEAN : CostReduction.NONE);
      costValue = cost.get();
    });
    return costValue;
  }

  predict(world: NDArray): Array2D {
    let values = null;
    this.math.scope(() => {
      const mapping =
          [{tensor: this.inputTensor, data: world.flatten().asType('float32')}];

      const evalOutput = this.session.eval(this.predictionTensor, mapping);
      values = evalOutput.dataSync();
    });
    return Array2D.new([this.size, this.size], values);
  }

  private setTrainingData(worlds: Array<[NDArray, NDArray]>): void {
    const inputs = [];
    const outputs = [];
    for (let i = 0; i < worlds.length; i++) {
      const example = worlds[i];
      inputs.push(example[0].flatten().asType('float32'));
      outputs.push(example[1].flatten().asType('float32'));
    }

    // TODO(kreeger): Don't really need to shuffle these.
    const inputProviderBuilder =
        new InGPUMemoryShuffledInputProviderBuilder([inputs, outputs]);
    const [inputProvider, targetProvider] =
        inputProviderBuilder.getInputProviders();

    this.feedEntries = [
      {tensor: this.inputTensor, data: inputProvider},
      {tensor: this.targetTensor, data: targetProvider}
    ];
  }

  /* Helper method for creating a fully connected layer. */
  private static createFullyConnectedLayer(
      graph: Graph, inputLayer: Tensor, layerIndex: number,
      sizeOfThisLayer: number, includeRelu = true, includeBias = true): Tensor {
    return graph.layers.dense(
        'fully_connected_' + layerIndex, inputLayer, sizeOfThisLayer,
        includeRelu ? (x) => graph.relu(x) : (x) => graph.sigmoid(x),
        includeBias);
  }

  /* Helper method for calculating loss. */
  private logLoss(graph: Graph, labelTensor: Tensor, predictionTensor: Tensor):
      Tensor {
    const epsilon = graph.constant(1e-7);
    const one = graph.constant(1);
    const negOne = graph.constant(-1);
    const predictionsPlusEps = graph.add(predictionTensor, epsilon);

    const left = graph.multiply(
        negOne, graph.multiply(labelTensor, graph.log(predictionsPlusEps)));
    const right = graph.multiply(
        graph.subtract(one, labelTensor),
        graph.log(graph.add(graph.subtract(one, predictionTensor), epsilon)));

    const losses = graph.subtract(left, right);
    const totalLosses = graph.reduceSum(losses);
    return graph.reshape(
        graph.divide(totalLosses, graph.constant(labelTensor.shape)), []);
  }
}
