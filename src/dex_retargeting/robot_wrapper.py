# typing.List：用于类型注解，表示“由 str 组成的列表”
from typing import List

# numpy：数值计算库
import numpy as np

# numpy.typing：为 numpy ndarray 提供类型注解支持
import numpy.typing as npt

# pinocchio：刚体动力学 / 运动学库（广泛用于机器人建模）
import pinocchio as pin


class RobotWrapper:
    """
    对 Pinocchio 的轻量封装类
    ⚠️ 当前实现不考虑 mimic joint（从动关节）
    """

    def __init__(self, urdf_path: str, use_collision=False, use_visual=False):
        # ------------------------------------------------------------------
        # 从 URDF 文件构建 Pinocchio 的机器人模型
        # Model：存储机器人拓扑结构、关节、惯量等“静态信息”
        # ------------------------------------------------------------------
        self.model: pin.Model = pin.buildModelFromUrdf(urdf_path)

        # Data：用于存储计算过程中产生的“中间变量”
        # 如：位姿、雅可比、动力学项等
        self.data: pin.Data = self.model.createData()

        # 当前封装不支持 collision / visual 几何
        # 若启用则直接抛异常
        if use_visual or use_collision:
            raise NotImplementedError

        # 计算机器人在“零位姿 / 中性位姿”下的关节配置
        # 对应 URDF 中的 reference pose
        self.q0 = pin.neutral(self.model)

        # 若速度自由度 nv != 位置自由度 nq
        # 说明存在特殊关节（如球关节、浮动基）
        # 当前实现无法处理
        if self.model.nv != self.model.nq:
            raise NotImplementedError("Can not handle robot with special joint.")

    # -------------------------------------------------------------------------- #
    # Robot property：机器人基本属性查询
    # -------------------------------------------------------------------------- #

    @property
    def joint_names(self) -> List[str]:
        """
        返回模型中所有 joint / frame 的名字
        注意：Pinocchio 的 names 包含 universe
        """
        return list(self.model.names)

    @property
    def dof_joint_names(self) -> List[str]:
        """
        返回“真正有自由度”的关节名字
        nqs[i] > 0 表示该 joint 在 configuration space 中有维度
        """
        nqs = self.model.nqs
        return [name for i, name in enumerate(self.model.names) if nqs[i] > 0]

    @property
    def dof(self) -> int:
        """
        机器人 configuration space 的维度
        等价于所有可动关节的自由度总和
        """
        return self.model.nq

    @property
    def link_names(self) -> List[str]:
        """
        返回所有 link（frame）的名字
        Pinocchio 使用 frame 统一表示 link / joint / visual
        """
        link_names = []
        for i, frame in enumerate(self.model.frames):
            link_names.append(frame.name)
        return link_names

    @property
    def joint_limits(self):
        """
        返回关节位置上下限
        shape = (nq, 2)
        """
        lower = self.model.lowerPositionLimit
        upper = self.model.upperPositionLimit
        return np.stack([lower, upper], axis=1)

    # -------------------------------------------------------------------------- #
    # Query function：索引 / 结构查询
    # -------------------------------------------------------------------------- #

    def get_joint_index(self, name: str):
        """
        根据关节名返回其在 q 向量中的索引
        ⚠️ 仅针对 dof_joint_names
        """
        return self.dof_joint_names.index(name)

    def get_link_index(self, name: str):
        """
        根据 link 名字返回 frame id
        """
        if name not in self.link_names:
            raise ValueError(
                f"{name} is not a link name. Valid link names: \n{self.link_names}"
            )

        # pin.BODY 表示只在 body frame 中查找
        return self.model.getFrameId(name, pin.BODY)

    def get_joint_parent_child_frames(self, joint_name: str):
        """
        给定 joint 名称，返回其父 link 和子 link 的 frame id
        """

        # joint 本身也是一个 frame
        joint_id = self.model.getFrameId(joint_name)

        # parent：该 joint 连接的父 frame
        parent_id = self.model.frames[joint_id].parent

        # child：通过 previousFrame 字段反向搜索
        child_id = -1
        for idx, frame in enumerate(self.model.frames):
            if frame.previousFrame == joint_id:
                child_id = idx

        if child_id == -1:
            raise ValueError(f"Can not find child link of {joint_name}")

        return parent_id, child_id

    # -------------------------------------------------------------------------- #
    # Kinematics function：运动学相关接口
    # -------------------------------------------------------------------------- #

    def compute_forward_kinematics(self, qpos: npt.NDArray):
        """
        执行正向运动学计算
        会更新 data 中所有 joint / frame 的位姿
        """
        pin.forwardKinematics(self.model, self.data, qpos)

    def get_link_pose(self, link_id: int) -> npt.NDArray:
        """
        获取指定 link 在世界坐标系下的齐次变换矩阵
        shape = (4, 4)
        """

        # 更新并返回 frame 的 SE3 位姿
        pose: pin.SE3 = pin.updateFramePlacement( #该链接相对于世界坐标系的 SE3 位姿
            self.model, self.data, link_id
        )

        # SE3 → 4x4 homogeneous matrix #齐次变换矩阵（4×4）
        return pose.homogeneous

    def get_link_pose_inv(self, link_id: int) -> npt.NDArray:
        """
        获取指定 link 位姿的逆变换
        常用于从 world → link 坐标系
        """

        pose: pin.SE3 = pin.updateFramePlacement(
            self.model, self.data, link_id
        )

        return pose.inverse().homogeneous

    def compute_single_link_local_jacobian(
        self, qpos, link_id: int
    ) -> npt.NDArray:
        """
        计算某一 link 的 Jacobian
        默认返回 6 x nq 的空间雅可比矩阵：
        [vx, vy, vz, wx, wy, wz]
        """

        J = pin.computeFrameJacobian(
            self.model, self.data, qpos, link_id
        )

        return J
