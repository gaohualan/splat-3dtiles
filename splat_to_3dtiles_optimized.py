import numpy as np
from pygltflib import GLTF2, Scene, Node, Mesh, Primitive, Buffer, BufferView, Accessor
import base64
import struct
import json
from pyproj import Transformer
from typing import List, Dict, Tuple
import argparse

is_geographic_coordinate = False

# 定义点的数据结构


class Point:
    def __init__(self, position: Tuple[float, float, float], color: Tuple[int, int, int, int],
                 scale: Tuple[float, float, float], rotation: Tuple[int, int, int, int]):
        self.position = position
        self.color = color
        self.scale = scale
        self.rotation = rotation

    def to_bytes(self) -> bytes:
        """将点数据打包为二进制格式"""
        return struct.pack('3f4B3f4B', *self.position, *self.color, *self.scale, *self.rotation)

    @classmethod
    def from_bytes(cls, data: bytes):
        """从二进制数据解析为点"""
        unpacked = struct.unpack('3f4B3f4B', data)
        position = unpacked[:3]
        color = unpacked[3:7]
        scale = unpacked[7:10]
        rotation = unpacked[10:]
        return cls(position, color, scale, rotation)

# 读取数据


def read_splat_file(file_path: str) -> List[Point]:
    """
    读取二进制格式的 Splat 文件
    :param file_path: Splat 文件路径
    :return: 包含位置、缩放、颜色、旋转数据的 Point 对象列表
    """
    points = []
    with open(file_path, 'rb') as f:
        while True:
            position_data = f.read(3 * 4)  # 3个 Float32，每个4字节
            if not position_data:
                break
            position = struct.unpack('3f', position_data)
            scale = struct.unpack('3f', f.read(3 * 4))
            color = struct.unpack('4B', f.read(4 * 1))
            rotation = struct.unpack('4B', f.read(4 * 1))
            points.append(Point(position, color, scale, rotation))
    return points

# 四叉树


class QuadTreeNode:
    def __init__(self, bounds: Tuple[float, float, float, float], capacity: int = 100000):
        """
        初始化四叉树节点。
        :param bounds: 节点的边界 (min_x, min_y, max_x, max_y)
        :param capacity: 节点容量（每个节点最多存储的点数）
        """
        self.bounds = bounds
        self.capacity = capacity
        self.points: List[Point] = []  # 存储点数据
        self.children = None

    def insert(self, point: Point) -> bool:
        """
        将点插入四叉树。
        :param point: 要插入的点
        :return: 是否插入成功
        """
        if not self._contains(point.position):
            return False

        if len(self.points) < self.capacity:
            self.points.append(point)
            return True
        else:
            if self.children is None:
                self._subdivide()
            return any(child.insert(point) for child in self.children)

    def _contains(self, position: Tuple[float, float, float]) -> bool:
        """
        检查点是否在节点边界内。
        :param position: 点的位置 (x, y, z)
        :return: 是否在边界内
        """
        x, y, _ = position
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

        for point in self.points:
            for child in self.children:
                if child.insert(point):
                    break

        self.points = []  # 清空当前节点的点数据

    def get_all_points(self) -> List[Point]:
        """
        获取当前节点及其子节点中的所有点。
        :return: 所有点的列表
        """
        points = self.points.copy()
        if self.children is not None:
            for child in self.children:
                points.extend(child.get_all_points())
        return points

# 将 Splat 数据转换为 glTF 文件


def splat_to_gltf_with_gaussian_extension(points: List[Point], output_path: str):
    """
    将 Splat 数据转换为支持 KHR_gaussian_splatting 扩展的 glTF 文件
    :param points: Point 对象列表
    :param output_path: 输出的 glTF 文件路径
    """
    # 提取数据
    positions = np.array(
        [point.position for point in points], dtype=np.float32)
    colors = np.array([point.color for point in points], dtype=np.uint8)
    scales = np.array([point.scale for point in points], dtype=np.float32)
    rotations = np.array([point.rotation for point in points], dtype=np.uint8)
    normalized_rotations = rotations / 255.0

    # 创建 GLTF 对象
    gltf = GLTF2()
    gltf.extensionsUsed = ["KHR_gaussian_splatting"]

    # 创建 Buffer
    buffer = Buffer()
    gltf.buffers.append(buffer)

    # 将数据转换为二进制
    positions_binary = positions.tobytes()
    colors_binary = colors.tobytes()
    scales_binary = scales.tobytes()
    rotations_binary = normalized_rotations.tobytes()

    # 创建 BufferView 和 Accessor
    def create_buffer_view(byte_offset: int, data: bytes, target: int = 34962) -> BufferView:
        return BufferView(buffer=0, byteOffset=byte_offset, byteLength=len(data), target=target)

    def create_accessor(buffer_view: int, component_type: int, count: int, type: str, max: List[float] = None, min: List[float] = None) -> Accessor:
        return Accessor(bufferView=buffer_view, componentType=component_type, count=count, type=type, max=max, min=min)

    buffer_views = [
        create_buffer_view(0, positions_binary),
        create_buffer_view(len(positions_binary), colors_binary),
        create_buffer_view(len(positions_binary) +
                           len(colors_binary), rotations_binary),
        create_buffer_view(len(positions_binary) +
                           len(colors_binary) + len(rotations_binary), scales_binary)
    ]
    accessors = [
        create_accessor(0, 5126, len(positions), "VEC3", positions.max(
            axis=0).tolist(), positions.min(axis=0).tolist()),
        create_accessor(1, 5121, len(colors), "VEC4"),
        create_accessor(2, 5126, len(normalized_rotations), "VEC4"),
        create_accessor(3, 5126, len(scales), "VEC3")
    ]
    gltf.bufferViews.extend(buffer_views)
    gltf.accessors.extend(accessors)

    # 创建 Mesh 和 Primitive
    primitive = Primitive(
        attributes={"POSITION": 0, "COLOR_0": 1, "_ROTATION": 2, "_SCALE": 3},
        mode=0,
        extensions={"KHR_gaussian_splatting": {
            "positions": 0, "colors": 1, "scales": 2, "rotations": 3}}
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
    gltf.buffers[0].uri = "data:application/octet-stream;base64," + base64.b64encode(
        positions_binary + colors_binary + rotations_binary + scales_binary).decode("utf-8")
    gltf.save(output_path)
    print(f"glTF 文件已保存到: {output_path}")

# 生成 3D Tiles


def generate_3dtiles(node: QuadTreeNode, output_dir: str, tile_name: str):
    if node.children is not None:
        for i, child in enumerate(node.children):
            generate_3dtiles(child, output_dir, f"{tile_name}_{i}")
    elif len(node.points) > 0:
        points = node.get_all_points()
        splat_to_gltf_with_gaussian_extension(
            points, f"{output_dir}/{tile_name}.gltf")

# 生成 tileset.json


def generate_tileset_json(output_dir: str, root_node: QuadTreeNode, bounds: List[float], geometric_error: int = 100):
    def build_tile_structure(node: QuadTreeNode, tile_name: str, current_geometric_error: int) -> Dict:
        bounding_volume = {"region": compute_region([point.position for point in node.get_all_points(
        )])} if is_geographic_coordinate else {"box": compute_box([point.position for point in node.get_all_points()])}
        content = {"uri": f"{tile_name}.gltf"} if not node.children else None
        children = [build_tile_structure(child, f"{tile_name}_{i}", current_geometric_error / 2)
                    for i, child in enumerate(node.children)] if node.children else []
        tile_structure = {
            "boundingVolume": bounding_volume,
            "geometricError": current_geometric_error,
            "refine": "ADD",
            "content": content
        }
        if children:
            tile_structure["children"] = children
            del tile_structure["content"]
        return tile_structure

    tileset = {
        "asset": {"version": "1.1", "gltfUpAxis": "Z"},
        "geometricError": geometric_error,
        "root": build_tile_structure(root_node, "tile_0", geometric_error)
    }
    with open(f"{output_dir}/tileset.json", "w") as f:
        json.dump(tileset, f, cls=NumpyEncoder, indent=4)

# 计算 box 范围


def compute_box(points: np.ndarray) -> List[float]:
    center = np.mean(points, axis=0)
    half_size = (np.max(points, axis=0) - np.min(points, axis=0)) / 2
    return [center[0], center[1], center[2], half_size[0], 0, 0, 0, half_size[1], 0, 0, 0, half_size[2]]

# 计算 region 范围


def compute_region(points: np.ndarray) -> List[float]:
    min_lon, min_lat, min_alt = np.min(points, axis=0)
    max_lon, max_lat, max_alt = np.max(points, axis=0)
    return [min_lon, min_lat, min_alt, max_lon, max_lat, max_alt]

# NumpyEncoder 用于序列化 NumPy 数组


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# 主函数


def main(input_path: str, output_dir: str):
    # 读取 .splat 文件
    points = read_splat_file(input_path)

    # 创建四叉树根节点
    positions = np.array([point.position for point in points])
    min_x, min_y = np.min(positions[:, :2], axis=0)
    max_x, max_y = np.max(positions[:, :2], axis=0)
    root = QuadTreeNode((min_x, min_y, max_x, max_y), capacity=100000)

    # 将点插入四叉树
    for point in points:
        root.insert(point)

    # 生成 3D Tiles
    generate_3dtiles(root, output_dir, "tile_0")

    # 生成 tileset.json
    bounds = [min_x, min_y, np.min(
        positions[:, 2]), max_x, max_y, np.max(positions[:, 2])]
    generate_tileset_json(output_dir, root, bounds)


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="将 Splat 文件转换为 3D Tiles。")
    parser.add_argument("input_path", type=str, help="输入的 .splat 文件路径")
    parser.add_argument("output_dir", type=str, help="输出的 3D Tiles 目录路径")
    args = parser.parse_args()

    # 调用主函数
    main(args.input_path, args.output_dir)
