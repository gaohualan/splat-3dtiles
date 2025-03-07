/**
 * @license
 * Cesium - https://github.com/CesiumGS/cesium
 * Version 1.125
 *
 * Copyright 2011-2022 Cesium Contributors
 *
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
 *
 * Columbus View (Pat. Pend.)
 *
 * Portions licensed separately.
 * See https://github.com/CesiumGS/cesium/blob/main/LICENSE.md for full licensing details.
 */

import{a as i,d as a}from"./chunk-AAXGYPN7.js";import{a as s}from"./chunk-4E3APMCC.js";import{e as o}from"./chunk-LRNH5AEO.js";async function c(r,n){let t=r.webAssemblyConfig;if(o(t)&&o(t.wasmBinary))return a(t.wasmBinary),!0}async function m(r,n){let t=r.webAssemblyConfig;if(o(t))return c(r,n);let{attributes:e,count:f}=r;return i(e.positions,e.scales,e.rotations,e.colors,f)}var y=s(m);export{y as default};
