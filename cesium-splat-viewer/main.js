let CesiumViewer;
let gaussianSplatTileset;

main();

async function main() {
  initViewer();
  loadTileset();
}

async function loadTileset() {

  try {
    const tileset = await Cesium.Cesium3DTileset.fromUrl(
      "http://localhost:8081/data/outputs/model/tileset.json",
      {
        modelMatrix:computeModelMatrix(),
        maximumScreenSpaceError: 1,
      }
    ).then((tileset) => {
      gaussianSplatTileset = tileset;
      CesiumViewer.scene.primitives.add(gaussianSplatTileset);
      setView();
    });
  } catch (error) {
    console.error(`Error creating tileset: ${error}`);
  }
}


function setView(){
  CesiumViewer.camera.flyTo({
    destination: Cesium.Cartesian3.fromDegrees(120, 30, 1),
    orientation: {
      heading: Cesium.Math.toRadians(30.0),
      pitch: Cesium.Math.toRadians(0),
      roll: 0.0,
    },
  });
}


function initViewer() {
  CesiumViewer = new Cesium.Viewer("cesiumContainer", {
    baseLayerPicker: false,
    baseLayer: Cesium.ImageryLayer.fromProviderAsync(
      Cesium.TileMapServiceImageryProvider.fromUrl(
        Cesium.buildModuleUrl("Assets/Textures/NaturalEarthII")
      )
    ),
    geocoder: false,
    timeline: false,
    animation: false,
    homeButton: false,
    fullscreenButton: false,
    selectionIndicator: false,
    infoBox: false,
    useDefaultRenderLoop: true,
    orderIndependentTranslucency: true,
    scene3DOnly: true,
    automaticallyTrackDataSourceClocks: false,
    dataSources: null,
    clock: null,
    targetFrameRate: 60,
    resolutionScale: 0.1,
    terrainShadows: Cesium.ShadowMode.ENABLED,
    navigationHelpButton: false,
    contextOptions: {
      webgl: {
        alpha: false,
        antialias: true,
        preserveDrawingBuffer: true,
        failIfMajorPerformanceCaveat: false,
        depth: false,
        stencil: true,
      },
    },
  });

  let utc = Cesium.JulianDate.fromDate(new Date("2025/02/06 04:00:00")); //UTC
  CesiumViewer.clockViewModel.currentTime = Cesium.JulianDate.addHours(
    utc,
    8,
    new Cesium.JulianDate()
  );

  CesiumViewer.scene.moon.show=false;
}

function computeModelMatrix() {
  const center = Cesium.Cartesian3.fromDegrees(
    120,
    30,
    10,
    CesiumViewer.scene.globe.ellipsoid
  );
  let modelMatrix = Cesium.Transforms.eastNorthUpToFixedFrame(center);

   const translationMatrix = Cesium.Matrix4.fromTranslation(
    Cesium.Cartesian3.fromArray([60, 100, 0.0])
  );

  modelMatrix = Cesium.Matrix4.multiply(
    modelMatrix,
    translationMatrix,
    new Cesium.Matrix4()
  );

  const rotationMatrix = Cesium.Matrix4.fromRotationTranslation(
    Cesium.Matrix3.fromRotationX(Cesium.Math.toRadians(-90))
  );

   modelMatrix = Cesium.Matrix4.multiply(
    modelMatrix,
    rotationMatrix,
    new Cesium.Matrix4()
  ); 

  const scaleMatrix = Cesium.Matrix4.fromScale(
    new Cesium.Cartesian3(6.0, 6.0, 6.0)
  );

  modelMatrix = Cesium.Matrix4.multiply(
    modelMatrix,
    scaleMatrix,
    new Cesium.Matrix4()
  ); 

  return modelMatrix;
}
