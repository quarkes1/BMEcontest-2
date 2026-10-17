import * as THREE from 'three';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';
import { MarchingCubes } from 'three/addons/objects/MarchingCubes.js';

type V = [number, number, number];
const smoothMin = (a:number,b:number,k:number) => {const h=Math.max(k-Math.abs(a-b),0)/k;return Math.min(a,b)-h*h*k*.25;};
function ellipsoid(x:number,y:number,z:number,c:V,r:V) {
  const a=(x-c[0])/r[0],b=(y-c[1])/r[1],d=(z-c[2])/r[2];
  const k0=Math.hypot(a,b,d),k1=Math.hypot(a/r[0],b/r[1],d/r[2]);
  return k0*(k0-1)/Math.max(k1,1e-8);
}
function capsule(x:number,y:number,z:number,a:V,b:V,ra:number,rb:number) {
  const dx=b[0]-a[0],dy=b[1]-a[1],dz=b[2]-a[2];
  const h=THREE.MathUtils.clamp(((x-a[0])*dx+(y-a[1])*dy+(z-a[2])*dz)/(dx*dx+dy*dy+dz*dz),0,1);
  return Math.hypot(x-a[0]-dx*h,y-a[1]-dy*h,z-a[2]-dz*h)-(ra+(rb-ra)*h);
}
/** One continuous implicit surface: palm pads, thumb web, and tapered relaxed digits. */
function handGeometry() {
  const n=88, mc=new MarchingCubes(n,new THREE.MeshBasicMaterial(),false,false,70000);
  const bones:{a:V;b:V;ra:number;rb:number}[]=[];
  // Little, ring, middle, index. Distal joints relax toward the palm.
  for(const [z,length,radius,spread] of [[-.365,.72,.105,-.09],[-.13,1.01,.12,-.025],[.13,1.13,.125,.015],[.375,.98,.12,.065]]) {
    const root:V=[1.08,.015,z],knuckle:V=[1.39,.01,z+spread*.25],joint:V=[1.39+length*.52,-.075,z+spread*.7],tip:V=[1.39+length,-.21,z+spread];
    bones.push({a:root,b:knuckle,ra:radius*1.25,rb:radius},{a:knuckle,b:joint,ra:radius,rb:radius*.88},{a:joint,b:tip,ra:radius*.88,rb:radius*.68});
  }
  bones.push({a:[.45,-.045,.28],b:[.76,-.10,.64],ra:.235,rb:.165},{a:[.76,-.10,.64],b:[1.09,-.18,.89],ra:.165,rb:.132},{a:[1.09,-.18,.89],b:[1.43,-.26,.94],ra:.132,rb:.105});
  for(let iz=0;iz<n;iz++)for(let iy=0;iy<n;iy++)for(let ix=0;ix<n;ix++) {
    const x=-.45+ix/n*3.45,y=-.9+iy/n*1.8,z=-1.25+iz/n*2.5;
    let d=ellipsoid(x,y,z,[.68,0,0],[.74,.225,.46]);
    d=smoothMin(d,ellipsoid(x,y,z,[.04,0,0],[.54,.245,.29]),.13);
    d=smoothMin(d,ellipsoid(x,y,z,[1.07,.012,0],[.36,.18,.48]),.12);
    for(const bone of bones)d=smoothMin(d,capsule(x,y,z,bone.a,bone.b,bone.ra,bone.rb),.075);
    mc.field[ix+iy*n+iz*n*n]=-d;
  }
  mc.isolation=0;mc.update();
  const count=mc.geometry.drawRange.count;
  const geometry=new THREE.BufferGeometry();
  for(const key of ['position','normal'])geometry.setAttribute(key,new THREE.Float32BufferAttribute((mc.geometry.getAttribute(key).array as Float32Array).slice(0,count*3),3));
  geometry.scale(1.725,.9,1.25);geometry.translate(1.275,0,0);geometry.computeBoundingSphere();
  mc.geometry.dispose();(mc.material as THREE.Material).dispose();return geometry;
}
function forearmGeometry(){
  const rings=90,radial=64,vertices:number[]=[],indices:number[]=[];
  for(let r=0;r<=rings;r++){
    const t=r/rings,x=-2.85+t*3.0;
    const end=Math.min(1,Math.sqrt(Math.max(.001,t/.055)));
    const ry=(.48-.245*t+.02*Math.sin(t*Math.PI))*end,rz=(.53-.245*t+.015*Math.sin(t*Math.PI))*end;
    for(let j=0;j<radial;j++){const a=j/radial*Math.PI*2;vertices.push(x,Math.sin(a)*ry,Math.cos(a)*rz);}
  }
  for(let r=0;r<rings;r++)for(let j=0;j<radial;j++){const a=r*radial+j,b=r*radial+(j+1)%radial,c=a+radial,d=b+radial;indices.push(a,c,b,b,c,d);}
  for(let j=1;j<radial-1;j++)indices.push(0,j,j+1);
  const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(vertices,3));geometry.setIndex(indices);geometry.computeVertexNormals();return geometry;
}
export function createForearmModel(){
  const group=new THREE.Group();group.name='Wrist IMU rigid body';
  const skin=new THREE.MeshStandardMaterial({color:'#dcdedb',roughness:.74,metalness:0});
  const strap=new THREE.MeshStandardMaterial({color:'#303638',roughness:.83});
  const metal=new THREE.MeshStandardMaterial({color:'#656e70',roughness:.36,metalness:.55});
  const glass=new THREE.MeshStandardMaterial({color:'#182225',roughness:.23,metalness:.12});
  group.add(new THREE.Mesh(forearmGeometry(),skin),new THREE.Mesh(handGeometry(),skin));
  const bandGeometry=new THREE.CylinderGeometry(1,1,.39,80,1,true);bandGeometry.rotateZ(Math.PI/2);
  bandGeometry.scale(1,.272,.315);bandGeometry.translate(-.11,0,0);
  const band=new THREE.Mesh(bandGeometry,strap);band.material.side=THREE.DoubleSide;group.add(band);
  const watch=new THREE.Group();watch.position.set(-.11,.281,0);group.add(watch);
  watch.add(new THREE.Mesh(new RoundedBoxGeometry(.54,.125,.65,6,.058),metal));
  const bezel=new THREE.Mesh(new RoundedBoxGeometry(.48,.028,.59,5,.065),strap);bezel.position.y=.073;watch.add(bezel);
  const screen=new THREE.Mesh(new RoundedBoxGeometry(.418,.016,.523,5,.055),glass);screen.position.y=.09;watch.add(screen);
  const crown=new THREE.Mesh(new THREE.CylinderGeometry(.035,.035,.045,24),metal);crown.rotation.x=Math.PI/2;crown.position.set(.10,.007,.344);watch.add(crown);
  const axis=new THREE.Group();axis.position.set(-.11,.42,0);group.add(axis);
  for(const [direction,color] of [[new THREE.Vector3(1,0,0),0xc5766e],[new THREE.Vector3(0,1,0),0x549b7c],[new THREE.Vector3(0,0,1),0x658abe]] as const)axis.add(new THREE.ArrowHelper(direction,new THREE.Vector3(),.48,color,.065,.026));
  group.userData.axis=axis;
  return {motionGroup:group,dispose:()=>{group.traverse(obj=>{if(obj instanceof THREE.Mesh||obj instanceof THREE.Line){obj.geometry.dispose();const materials=Array.isArray(obj.material)?obj.material:[obj.material];materials.forEach(m=>m.dispose());}});}};
}

