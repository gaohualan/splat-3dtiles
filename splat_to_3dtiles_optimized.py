import numpy as np
from pygltflib import GLTF2, Scene, Node, Mesh, Primitive, Buffer, BufferView, Accessor
import base64
import struct
import json
from pyproj import Transformer

is_geographic_coordinate=False

#读取数据
def read_splat_file(file_path):

    """
    读取二进制格式的 Splat 文件
    :param file_path: Splat 文件路径
    :return: 包含位置、缩放、颜色、旋转数据的字典
    """
    with open(file_path, 'rb') as f:
        # 初始化存储数据的列表
        positions = []
        scales = []
        colors = []
        rotations = []

        # 逐点读取数据
        while True:
            # 读取位置（3个 Float32，x, y, z）
            position_data = f.read(3 * 4)  # 3个 Float32，每个4字节
            if not position_data:
                break  # 文件结束
            x, y, z = struct.unpack('3f', position_data)
            positions.append([x, y, z])

            # 读取缩放（3个 Float32，sx, sy, sz）
            scale_data = f.read(3 * 4)  # 3个 Float32，每个4字节
            sx, sy, sz = struct.unpack('3f', scale_data)
            scales.append([sx, sy, sz])

            # 读取颜色（4个 uint8，r, g, b, a）
            color_data = f.read(4 * 1)  # 4个 uint8，每个1字节
            r, g, b, a = struct.unpack('4B', color_data)
            colors.append([r, g, b, a])
            #print("r, g, b, a:",[r, g, b, a])

            # 读取旋转（4个 uint8，i, j, k, l）
            rotation_data = f.read(4 * 1)  # 4个 uint8，每个1字节
            i, j, k, l = struct.unpack('4B', rotation_data)
            rotations.append([i, j, k, l])

        # 将列表转换为 NumPy 数组
        positions = np.array(positions, dtype=np.float32)
        scales = np.array(scales, dtype=np.float32)
        colors = np.array(colors, dtype=np.uint8)
        rotations = np.array(rotations, dtype=np.uint8)

        # 返回解析的数据
        return {
            'positions': positions,
            'scales': scales,
            'colors': colors,
            'rotations': rotations
        }

#四叉树
class QuadTreeNode:
    def __init__(self, bounds, capacity=100000):
        """
        初始化四叉树节点。
        :param bounds: 节点的边界 (min_x, min_y, max_x, max_y)
        :param capacity: 节点容量（每个节点最多存储的点数）
        """
        self.bounds = bounds
        self.capacity = capacity
        self.points = []
        self.children = None
        self.colors = []
        self.scales = []
        self.rotations = []

    def insert(self, point,color,scale,rotation):
        """
        将点插入四叉树。
        """
        if not self._contains(point):
            return False

        if len(self.points) < self.capacity:
            self.points.append(point)
            self.colors.append(color)
            self.scales.append(scale)
            self.rotations.append(rotation)
            return True
        else:
            if self.children is None:
                self._subdivide()

            for child in self.children:
                if child.insert(point,color,scale,rotation):
                    return True

        return False

    def _contains(self, point):
        """
        检查点是否在节点边界内。
        """
        x, y, _ = point
        min_x, min_y, max_x, max_y = self.bounds
        return min_x <= x < max_x and min_y <= y < max_y

    def _subdivide(self):
        """
        将节点划分为四个子节点。
        """
        min_x, min_y, max_x, max_y = self.bounds
        mid_x = (min_x + max_x) / 2
        mid_y = (min_y + max_y) / 2

        self.children = [
            QuadTreeNode((min_x, min_y, mid_x, mid_y), self.capacity),
            QuadTreeNode((mid_x, min_y, max_x, mid_y), self.capacity),
            QuadTreeNode((min_x, mid_y, mid_x, max_y), self.capacity),
            QuadTreeNode((mid_x, mid_y, max_x, max_y), self.capacity)
        ]

        for index, point in enumerate(self.points):
            for child in self.children:
                if child.insert(point,self.colors[index],self.scales[index],self.rotations[index]):
                    break

        self.points = []
        self.colors = []
        self.scales = []
        self.rotations = []


def splat_to_gltf_with_gaussian_extension(positions, colors, scales, rotations, output_path):
    """
    将 Splat 数据转换为支持 KHR_gaussian_splatting 扩展的 glTF 文件
    :param positions: 位置数据（Nx3 的浮点数数组）
    :param colors: 颜色数据（Nx4 的 uint8 数组，RGBA）
    :param scales: 缩放数据（Nx3 的浮点数数组）
    :param rotations: 旋转数据（Nx4 的 uint8 数组，IJKL 四元数）
    :param output_path: 输出的 glTF 文件路径
    """
    # 将颜色和旋转数据归一化到 [0, 1] 范围
    #normalized_colors = colors / 255.0
    normalized_rotations = rotations / 255.0

    # 创建 GLTF 对象
    gltf = GLTF2()

    # 添加 KHR_gaussian_splatting 扩展
    gltf.extensionsUsed = [
        #"KHR_materials_unlit",
        "KHR_gaussian_splatting"
    ]
    #gltf.extensionsUsed = ["KHR_gaussian_splatting"]
    #gltf.extensionsRequired = ["KHR_gaussian_splatting"]

    # 创建 Buffer
    buffer = Buffer()
    gltf.buffers.append(buffer)

    # 将数据转换为二进制
    positions_binary = positions.tobytes()
    colors_binary = colors.tobytes() #normalized_colors.tobytes()
    scales_binary = scales.tobytes()
    rotations_binary = normalized_rotations.tobytes()

    # 创建 BufferView 和 Accessor 用于位置数据
    positions_buffer_view = BufferView(
        buffer=0,
        byteOffset=0,
        byteLength=len(positions_binary),
        target=34962  # ARRAY_BUFFER
    )
    positions_accessor = Accessor(
        bufferView=0,
        componentType=5126,  # FLOAT
        count=len(positions),
        type="VEC3",
        max=positions.max(axis=0).tolist(),
        min=positions.min(axis=0).tolist()
    )
    gltf.bufferViews.append(positions_buffer_view)
    gltf.accessors.append(positions_accessor)

    # 创建 BufferView 和 Accessor 用于颜色数据
    colors_buffer_view = BufferView(
        buffer=0,
        byteOffset=len(positions_binary),
        byteLength=len(colors_binary),
        target=34962  # ARRAY_BUFFER
    )
    colors_accessor = Accessor(
        bufferView=1,
        componentType=5121,  # UNSIGNED BYTE
        count=len(colors), #,len(normalized_colors),#
        type="VEC4"
    )
    gltf.bufferViews.append(colors_buffer_view)
    gltf.accessors.append(colors_accessor)

 # 创建 BufferView 和 Accessor 用于旋转数据
    rotations_buffer_view = BufferView(
        buffer=0,
        byteOffset=len(positions_binary) + len(colors_binary) ,
        byteLength=len(rotations_binary),
        target=34962  # ARRAY_BUFFER
    )
    rotations_accessor = Accessor(
        bufferView=2,
        componentType=5126,  # FLOAT
        count=len(normalized_rotations),
        type="VEC4"
    )
    gltf.bufferViews.append(rotations_buffer_view)
    gltf.accessors.append(rotations_accessor)

    # 创建 BufferView 和 Accessor 用于缩放数据
    scales_buffer_view = BufferView(
        buffer=0,
        byteOffset=len(positions_binary) + len(colors_binary)+ len(rotations_binary),
        byteLength=len(scales_binary),
        target=34962  # ARRAY_BUFFER
    )
    scales_accessor = Accessor(
        bufferView=3,
        componentType=5126,  # FLOAT
        count=len(scales),
        type="VEC3"
    )
    gltf.bufferViews.append(scales_buffer_view)
    gltf.accessors.append(scales_accessor)

   

    # 创建 Mesh 和 Primitive
    primitive = Primitive(
        attributes={
            "POSITION": 0,
            "COLOR_0": 1,
            "_ROTATION": 2,
            "_SCALE": 3
        },
        mode=0,  # POINTS
        extensions={
            "KHR_gaussian_splatting": {
                "positions": 0,
                "colors": 1,
                "scales": 2,
                "rotations": 3
            }
        }
    )
    mesh = Mesh(primitives=[primitive])
    gltf.meshes.append(mesh)

    # 创建 Node 和 Scene
    node = Node(mesh=0)
    gltf.nodes.append(node)
    scene = Scene(nodes=[0])
    gltf.scenes.append(scene)
    gltf.scene = 0

    # 将二进制数据写入 Buffer
    gltf.buffers[0].uri = "data:application/octet-stream;base64," + \
        base64.b64encode(positions_binary + colors_binary + rotations_binary + scales_binary ).decode("utf-8")

    # 保存为 glTF 文件
    gltf.save(output_path)
    print(f"glTF 文件已保存到: {output_path}")



def generate_3dtiles(node, output_dir, tile_name):
    """
    递归生成 3D Tiles。
    """
    if node.children is not None:
        for i, child in enumerate(node.children):
            generate_3dtiles(child, output_dir, f"{tile_name}_{i}")
    else:
        if len(node.points) > 0:
            points = np.array(node.points)
            colors = np.array(node.colors) 
            scales = np.array(node.scales)
            rotations = np.array(node.rotations)

            splat_to_gltf_with_gaussian_extension(points, colors,scales,rotations, f"{output_dir}/{tile_name}.gltf")
            #tile = create_pnts_tile(points, colors)
            #tile.save_to(f"{output_dir}/{tile_name}.gltf")

# 计算 Transform
def compute_transform(lon, lat, alt):
    """
    计算从局部坐标系到 WGS84 坐标系的变换矩阵。
    :param lon: 局部坐标系原点的经度
    :param lat: 局部坐标系原点的纬度
    :param alt: 局部坐标系原点的高度
    :return: 4x4 变换矩阵 (列优先顺序的 16 个元素)
    """
    # 将经纬度转换为弧度
    lon_rad = np.radians(lon)
    lat_rad = np.radians(lat)

    # 计算旋转矩阵（将局部坐标系对齐到 WGS84）
    cos_lon = np.cos(lon_rad)
    sin_lon = np.sin(lon_rad)
    cos_lat = np.cos(lat_rad)
    sin_lat = np.sin(lat_rad)

    # 旋转矩阵
    rotation = np.array([
        [-sin_lon, -sin_lat * cos_lon, cos_lat * cos_lon, 0],
        [cos_lon, -sin_lat * sin_lon, cos_lat * sin_lon, 0],
        [0, cos_lat, sin_lat, 0],
        [0, 0, 0, 1]
    ])

    # 平移矩阵
    translation = np.array([
        [1, 0, 0, lon],
        [0, 1, 0, lat],
        [0, 0, 1, alt],
        [0, 0, 0, 1]
    ])

    # 变换矩阵 = 平移矩阵 @ 旋转矩阵
    transform = translation @ rotation

    # 将矩阵展平为列优先顺序的 16 个元素
    return transform.T.flatten()

# 坐标转换
def transform_coordinates(points, source_crs, target_crs='EPSG:4326'):
    """
    将点云数据从局部坐标系转换为 WGS84 坐标系。
    :param points: 点云数据 (Nx3 数组)
    :param source_crs: 源坐标系 (例如 'EPSG:3857')
    :param target_crs: 目标坐标系 (默认 'EPSG:4326'，即 WGS84)
    :return: 转换后的点云数据
    """
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    x, y = transformer.transform(points[:, 0], points[:, 1])
    return np.column_stack((x, y, points[:, 2]))

class NumpyEncoder(json.JSONEncoder):  
    def default(self, obj):  
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,  
            np.int16, np.int32, np.int64, np.uint8,  
            np.uint16, np.uint32, np.uint64)):  
            return int(obj)  
        elif isinstance(obj, (np.float_, np.float16, np.float32,np.float64)):  
            return float(obj)  
        elif isinstance(obj, (np.ndarray,)):  
            return obj.tolist()  
        return json.JSONEncoder.default(self, obj)


def compute_box(points):
    """
    计算点云数据的 box 范围。
    :param points: 点云数据 (Nx3 数组，每行是 [x, y, z])
    :return: box 范围 [centerX, centerY, centerZ, xDirX, xDirY, xDirZ, yDirX, yDirY, yDirZ, zDirX, zDirY, zDirZ]
    """
    # 计算中心点
    center = np.mean(points, axis=0)

    # 计算半长向量
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    half_size = (max_coords - min_coords) / 2

    # 构造 box
    box = [
        center[0], center[1], center[2],  # 中心点
        half_size[0], 0, 0,               # X 轴方向
        0, half_size[1], 0,               # Y 轴方向
        0, 0, half_size[2]                # Z 轴方向
    ]
    return box


def compute_region(points):
    """
    计算点云数据的 region 范围。
    :param points: 点云数据 (Nx3 数组，每行是 [经度, 纬度, 高度])
    :return: region 范围 [min_longitude, min_latitude, min_height, max_longitude, max_latitude, max_height]
    """
    min_lon, min_lat, min_alt = np.min(points, axis=0)
    max_lon, max_lat, max_alt = np.max(points, axis=0)
    return [min_lon, min_lat, min_alt, max_lon, max_lat, max_alt]

        
# 生成tileset.json
def generate_tileset_json(output_dir, root_node, bounds, geometric_error=100):
    """
    递归生成符合 3D Tiles 1.1 规范的 tileset.json 文件。
    :param output_dir: 输出目录
    :param root_node: 四叉树的根节点
    :param bounds: 根节点的边界 (min_x, min_y, min_z, max_x, max_y, max_z)
    :param geometric_error: 几何误差
    """
    def build_tile_structure(node, tile_name, current_geometric_error):
        """
        递归构建 Tile 结构。
        :param node: 当前四叉树节点
        :param tile_name: 当前 Tile 的名称
        :param current_geometric_error: 当前 Tile 的几何误差
        :return: 当前 Tile 的结构
        """
        # 当前 Tile 的 boundingVolume
        # 计算 boundingVolume
        if is_geographic_coordinate:  # 如果是地理坐标系
          bounding_volume = {
              "region": compute_region(node.points)
          }
        else:  # 如果是局部坐标系
          bounding_volume = {
            "box": compute_box(node.points)
          }

        # 当前 Tile 的内容
        content = {
            "uri": f"{tile_name}.gltf"
        }

        # 子节点列表
        children = []
        if node.children is not None:
            for i, child in enumerate(node.children):
                child_tile_name = f"{tile_name}_{i}"
                children.append(
                    build_tile_structure(child, child_tile_name, current_geometric_error / 2)
                )

        # 当前 Tile 的结构
        tile_structure = {
            "boundingVolume": bounding_volume,
            "geometricError": current_geometric_error,
            "refine": "ADD",  # 细化方式
            "content": content
        }

        # 如果有子节点，则添加到 children 中
        if children:
            tile_structure["children"] = children
            #tile_structure["content"] = ""
            del tile_structure["content"]

        return tile_structure

    # 构建根节点的 Tile 结构
    root_tile_structure = build_tile_structure(root_node, "tile_0", geometric_error)

    # tileset 结构
    tileset = {
        "asset": {
            "version": "1.1",
            "gltfUpAxis": "Z"  # 默认 Z 轴向上
        },
        "geometricError": geometric_error,
        "root": root_tile_structure
    }

    # 写入文件
    with open(f"{output_dir}/tileset.json", "w") as f:
        json.dump(tileset, f,cls=NumpyEncoder, indent=4)



def main():
    input_path=r"D:\data\splat\train.splat"
    output_dir = r'D:\code\test\py3dtiles\cesium-splat-viewer\data\outputs\train'

    # 读取 .splat 文件
    splat_data=read_splat_file(input_path)
    positions=splat_data['positions']
    scales=splat_data['scales']
    colors=splat_data['colors']
    rotations=splat_data['rotations']
    points=positions

    # 坐标转换（如果需要）
    #source_crs = 'EPSG:3857'  # 假设源坐标系是 Web Mercator
    #points = transform_coordinates(points, source_crs)
    
    # 创建四叉树根节点
    #min_x, min_y = np.min(points[:, :2], axis=0)
    #max_x, max_y = np.max(points[:, :2], axis=0)
    min_x = np.min(points[:, 0], axis=0)
    min_y = np.min(points[:, 1], axis=0)
    max_x = np.max(points[:, 0], axis=0)
    max_y = np.max(points[:, 1], axis=0)
    root = QuadTreeNode((min_x, min_y, max_x, max_y),100000)

    # 将点插入四叉树
    for index, point in enumerate(points):
        root.insert(point,colors[index],scales[index],rotations[index])

    # 生成 3D Tiles   
    generate_3dtiles(root, output_dir, "tile_0")

    # 生成 tileset.json
    bounds = [min_x, min_y, np.min(points[:, 2]), max_x, max_y, np.max(points[:, 2])]
    generate_tileset_json(output_dir, root, bounds)

if __name__ == "__main__":
    main()