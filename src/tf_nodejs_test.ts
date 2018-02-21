import {ENV, Environment, tensor2d, Tensor2D} from '.';

Environment.setBackend('nodejs');
ENV.engine.startScope();

const t1: Tensor2D = tensor2d([[1, 2], [3, 4]]);
const t2: Tensor2D = tensor2d([[5, 6], [7, 8]]);
const result = ENV.math.matMul(t1, t2);
const padded = ENV.math.pad2D(result, [[1, 1], [1, 1]]);

console.log('t1', t1.dataSync());
console.log('t2', t2.dataSync());
console.log('matmul: ', result.dataSync());
console.log('padded: ', padded.dataSync());

ENV.engine.endScope(null);
