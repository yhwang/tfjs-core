import {ENV, Environment, Tensor, tensor1d} from '.';

Environment.setBackend('nodejs');
ENV.engine.startScope();

console.log('hi');
const t1: Tensor = tensor1d([1, 2, 3]);
const t2: Tensor = tensor1d([3, 4, 5]);

const result = ENV.math.equal(t1, t2);
console.log('result', result);

ENV.engine.endScope(null);
