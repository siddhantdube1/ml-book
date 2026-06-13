import { trainPerceptron, trainMLP, mlpForward, hiddenActivations, makeXORData, makeMoonsData, makeCirclesData, sigmoid, type Point, type MLP } from './lib/neuralnet'
import { createRng, gauss } from './lib/rng'

// ── gradient check (finite differences) on a tiny net ──
console.log('=== gradient check ===')
const rng = createRng(1)
const tinyData: Point[] = [{x:[0.5,-0.3],y:1},{x:[-0.7,0.2],y:0},{x:[0.1,0.9],y:1},{x:[-0.4,-0.6],y:0}]
function makeNet(): MLP {
  return { W:[[[gauss(rng,0,0.7),gauss(rng,0,0.7)],[gauss(rng,0,0.7),gauss(rng,0,0.7)],[gauss(rng,0,0.7),gauss(rng,0,0.7)]],[[gauss(rng,0,0.7),gauss(rng,0,0.7),gauss(rng,0,0.7)]]], b:[[0,0,0],[0]], act:'tanh' }
}
function bce(net:MLP, data:Point[]){let l=0;for(const p of data){const o=mlpForward(net,p.x).output;l+=-(p.y*Math.log(o+1e-9)+(1-p.y)*Math.log(1-o+1e-9))}return l/data.length}
const net = makeNet()
// analytic grad via one training step diff: instead, compare to finite diff on W[0][0][0]
const eps=1e-5
function fdGrad(net:MLP, li:number, i:number, j:number){const up=JSON.parse(JSON.stringify(net));up.W[li][i][j]+=eps;const dn=JSON.parse(JSON.stringify(net));dn.W[li][i][j]-=eps;return (bce(up,tinyData)-bce(dn,tinyData))/(2*eps)}
// analytic: run trainMLP for 1 epoch lr=1 and read the applied gradient? Instead recompute backprop inline:
function analyticGrad(net:MLP, data:Point[]){
  const L=net.W.length; const gW=net.W.map(m=>m.map(r=>r.map(()=>0)))
  for(const p of data){
    // forward
    const a:number[][]=[p.x as number[]]; const z:number[][]=[]
    for(let l=0;l<L;l++){const prev=a[l];const zl=net.W[l].map((row,i)=>{let s=net.b[l][i];for(let k=0;k<row.length;k++)s+=row[k]*prev[k];return s});z.push(zl);const isOut=l===L-1;a.push(zl.map(v=>isOut?sigmoid(v):Math.tanh(v)))}
    let delta=[a[L][0]-p.y]
    for(let l=L-1;l>=0;l--){for(let i=0;i<delta.length;i++)for(let j=0;j<a[l].length;j++)gW[l][i][j]+=delta[i]*a[l][j]
      if(l>0){const nin=net.W[l][0].length;const nd=new Array(nin).fill(0);for(let i=0;i<delta.length;i++)for(let j=0;j<nin;j++)nd[j]+=net.W[l][i][j]*delta[i];for(let j=0;j<nin;j++)nd[j]*=(1-Math.tanh(z[l-1][j])**2);delta=nd}}
  }
  return gW.map(m=>m.map(r=>r.map(v=>v/data.length)))
}
const ag = analyticGrad(net, tinyData)
console.log('W[0][0][0] analytic:', ag[0][0][0].toFixed(6), 'finite-diff:', fdGrad(net,0,0,0).toFixed(6))
console.log('W[1][0][1] analytic:', ag[1][0][1].toFixed(6), 'finite-diff:', fdGrad(net,1,0,1).toFixed(6))

// ── perceptron: converges on separable, thrashes on XOR ──
console.log('\n=== perceptron ===')
const sep: Point[] = []
const r2 = createRng(4)
for(let i=0;i<60;i++){const y=i%2 as 0|1; sep.push({x:[gauss(r2,y?1.6:-1.6,0.5),gauss(r2,y?0.6:-0.6,0.5)],y})}
const fp = trainPerceptron(sep, 50, 0.1)
console.log('separable: frames', fp.length, 'final errors', fp[fp.length-1].errors, fp[fp.length-1].errors===0?'(converged)':'')
const fx = trainPerceptron(makeXORData(80,3), 50, 0.1)
console.log('XOR: frames', fx.length, 'final errors', fx[fx.length-1].errors, '(never 0 -> thrash)')

// ── MLP trains to ~100% on XOR/moons/circles ──
console.log('\n=== MLP training ===')
for (const [name, data, hidden, lr, epochs] of [
  ['XOR', makeXORData(160,3), [4], 0.8, 400],
  ['moons', makeMoonsData(160,5), [8], 0.5, 600],
  ['circles', makeCirclesData(160,5), [8], 0.5, 600],
] as [string, Point[], number[], number, number][]) {
  const t0=performance.now()
  const frames = trainMLP(data, hidden, {epochs, lr, act:'tanh', seed:2})
  const t1=performance.now()
  const last = frames[frames.length-1]
  console.log(`${name.padEnd(8)} hidden=${JSON.stringify(hidden)} -> acc=${(last.accuracy*100).toFixed(1)}% loss=${last.loss.toFixed(3)} frames=${frames.length} ${(t1-t0).toFixed(0)}ms`)
}

// ── hidden representation makes XOR linearly separable ──
console.log('\n=== hidden representation ===')
const xor = makeXORData(160, 3)
const frames = trainMLP(xor, [2], {epochs: 800, lr: 0.7, act:'tanh', seed: 7})
const trained = frames[frames.length-1]
console.log('2-hidden XOR net acc:', (trained.accuracy*100).toFixed(1)+'%')
// train a linear classifier on hidden activations -> should be ~100%
const H = xor.map(p => ({ h: hiddenActivations(trained.net, p.x, 0), y: p.y }))
// simple logistic on 2D hidden
let lw=[0,0], lb=0
for(let it=0;it<500;it++){let gw=[0,0],gb=0;for(const s of H){const z=lw[0]*s.h[0]+lw[1]*s.h[1]+lb;const o=sigmoid(z);const e=o-s.y;gw[0]+=e*s.h[0];gw[1]+=e*s.h[1];gb+=e}lw[0]-=0.5*gw[0]/H.length;lw[1]-=0.5*gw[1]/H.length;lb-=0.5*gb/H.length}
let lin=0;for(const s of H){const z=lw[0]*s.h[0]+lw[1]*s.h[1]+lb;if((z>=0?1:0)===s.y)lin++}
console.log('linear classifier on HIDDEN activations:', (lin/H.length*100).toFixed(1)+'% (input space was not separable)')
