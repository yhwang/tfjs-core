import {ENV, Environment, Tensor, tensor1d} from '.';

Environment.setBackend('nodejs');
ENV.engine.startScope();

const t1: Tensor = tensor1d([1, 2, 3]);
const t2: Tensor = tensor1d([3, 2, 5]);
console.log('t1', t1.dataSync());
console.log('t2', t2.dataSync());

const result = ENV.math.equal(t1, t2);
console.log('equal: ', result.data());
ENV.engine.endScope(null);
