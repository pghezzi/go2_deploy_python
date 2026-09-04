from dataclasses import dataclass
import cyclonedds.idl as idl
import cyclonedds.idl.annotations as annotate
import cyclonedds.idl.types as types


@dataclass
@annotate.final
@annotate.autoid("sequential")
class DepthImage_(idl.IdlStruct, typename="unitree_go::msg::dds_::DepthImage_"):
    width: types.uint16
    height: types.uint16
    normalized_value: types.sequence[types.float32]