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

import{b as f,c as o,d as u}from"./chunk-AAXGYPN7.js";import{a}from"./chunk-4E3APMCC.js";import{e as r}from"./chunk-LRNH5AEO.js";async function m(i,s){let t=i.webAssemblyConfig;if(r(t)&&r(t.wasmBinary))return u(t.wasmBinary),!0}var c=2048;function d(i,s){let t=i.webAssemblyConfig;if(r(t))return m(i,s);let{primitive:e,sortType:n}=i;if(n==="Attribute")return f(e.attributes,e.modelView,e.count);if(n==="Index")return o(e.positions,e.modelView,c,e.count);if(n==="SIMD Index")return o(e.positions,e.modelView,c,e.count)}var g=a(d);export{g as default};
