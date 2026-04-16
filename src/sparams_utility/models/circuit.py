from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class FrequencySweepSpec:
    fmin_hz: float = 1e7
    fmax_hz: float = 1e10
    fstep_hz: float = 1e7
    display_unit: str = "GHz"

    def to_dict(self) -> dict:
        return {
            "fmin_hz": self.fmin_hz,
            "fmax_hz": self.fmax_hz,
            "fstep_hz": self.fstep_hz,
            "display_unit": self.display_unit,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "FrequencySweepSpec":
        return cls(
            fmin_hz=float(payload.get("fmin_hz", 1e7)),
            fmax_hz=float(payload.get("fmax_hz", 1e10)),
            fstep_hz=float(payload.get("fstep_hz", 1e7)),
            display_unit=str(payload.get("display_unit", "GHz")),
        )


@dataclass(frozen=True)
class CircuitPortRef:
    instance_id: str
    port_number: int

    def key(self) -> Tuple[str, int]:
        return self.instance_id, self.port_number

    def to_dict(self) -> dict:
        return {
            "instance_id": self.instance_id,
            "port_number": self.port_number,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "CircuitPortRef":
        return cls(
            instance_id=str(payload.get("instance_id", "")),
            port_number=int(payload.get("port_number", 0)),
        )


@dataclass(frozen=True)
class CircuitBlockInstance:
    instance_id: str
    source_file_id: str
    display_label: str
    nports: int
    position_x: float
    position_y: float
    block_kind: str = "touchstone"
    impedance_ohm: float = 50.0
    rotation_deg: int = 0
    mirror_horizontal: bool = False
    mirror_vertical: bool = False

    def to_dict(self) -> dict:
        return {
            "instance_id": self.instance_id,
            "source_file_id": self.source_file_id,
            "display_label": self.display_label,
            "nports": self.nports,
            "position_x": self.position_x,
            "position_y": self.position_y,
            "block_kind": self.block_kind,
            "impedance_ohm": self.impedance_ohm,
            "rotation_deg": self.rotation_deg,
            "mirror_horizontal": self.mirror_horizontal,
            "mirror_vertical": self.mirror_vertical,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "CircuitBlockInstance":
        return cls(
            instance_id=str(payload.get("instance_id", "")),
            source_file_id=str(payload.get("source_file_id", "")),
            display_label=str(payload.get("display_label", "Block")),
            nports=int(payload.get("nports", 0)),
            position_x=float(payload.get("position_x", 0.0)),
            position_y=float(payload.get("position_y", 0.0)),
            block_kind=str(payload.get("block_kind", "touchstone")),
            impedance_ohm=float(payload.get("impedance_ohm", 50.0)),
            rotation_deg=int(payload.get("rotation_deg", 0)),
            mirror_horizontal=bool(payload.get("mirror_horizontal", False)),
            mirror_vertical=bool(payload.get("mirror_vertical", False)),
        )


@dataclass(frozen=True)
class CircuitConnection:
    connection_id: str
    port_a: CircuitPortRef
    port_b: CircuitPortRef

    def to_dict(self) -> dict:
        return {
            "connection_id": self.connection_id,
            "port_a": self.port_a.to_dict(),
            "port_b": self.port_b.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "CircuitConnection":
        return cls(
            connection_id=str(payload.get("connection_id", "")),
            port_a=CircuitPortRef.from_dict(payload.get("port_a", {})),
            port_b=CircuitPortRef.from_dict(payload.get("port_b", {})),
        )


@dataclass(frozen=True)
class ExternalPortAssignment:
    external_port_number: int
    port_ref: CircuitPortRef

    def to_dict(self) -> dict:
        return {
            "external_port_number": self.external_port_number,
            "port_ref": self.port_ref.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "ExternalPortAssignment":
        return cls(
            external_port_number=int(payload.get("external_port_number", 0)),
            port_ref=CircuitPortRef.from_dict(payload.get("port_ref", {})),
        )


@dataclass(frozen=True)
class CircuitValidationIssue:
    message: str


@dataclass
class CircuitDocument:
    instances: List[CircuitBlockInstance] = field(default_factory=list)
    connections: List[CircuitConnection] = field(default_factory=list)
    external_ports: List[ExternalPortAssignment] = field(default_factory=list)
    sweep: FrequencySweepSpec = field(default_factory=FrequencySweepSpec)

    def next_instance_id(self) -> str:
        used = {item.instance_id for item in self.instances}
        index = 1
        while f"inst-{index}" in used:
            index += 1
        return f"inst-{index}"

    def next_connection_id(self) -> str:
        used = {item.connection_id for item in self.connections}
        index = 1
        while f"conn-{index}" in used:
            index += 1
        return f"conn-{index}"

    def add_instance(
        self,
        *,
        source_file_id: str,
        display_label: str,
        nports: int,
        position_x: float,
        position_y: float,
        block_kind: str = "touchstone",
        impedance_ohm: float = 50.0,
        rotation_deg: int = 0,
        mirror_horizontal: bool = False,
        mirror_vertical: bool = False,
    ) -> CircuitBlockInstance:
        instance = CircuitBlockInstance(
            instance_id=self.next_instance_id(),
            source_file_id=source_file_id,
            display_label=display_label,
            nports=nports,
            position_x=position_x,
            position_y=position_y,
            block_kind=block_kind,
            impedance_ohm=impedance_ohm,
            rotation_deg=rotation_deg,
            mirror_horizontal=mirror_horizontal,
            mirror_vertical=mirror_vertical,
        )
        self.instances.append(instance)
        self.rebuild_external_ports_from_instances()
        return instance

    def update_instance_position(self, instance_id: str, x: float, y: float) -> None:
        for index, instance in enumerate(self.instances):
            if instance.instance_id != instance_id:
                continue
            self.instances[index] = CircuitBlockInstance(
                instance_id=instance.instance_id,
                source_file_id=instance.source_file_id,
                display_label=instance.display_label,
                nports=instance.nports,
                position_x=x,
                position_y=y,
                block_kind=instance.block_kind,
                impedance_ohm=instance.impedance_ohm,
                rotation_deg=instance.rotation_deg,
                mirror_horizontal=instance.mirror_horizontal,
                mirror_vertical=instance.mirror_vertical,
            )
            return

    def update_instance_impedance(self, instance_id: str, impedance_ohm: float) -> None:
        for index, instance in enumerate(self.instances):
            if instance.instance_id != instance_id:
                continue
            self.instances[index] = CircuitBlockInstance(
                instance_id=instance.instance_id,
                source_file_id=instance.source_file_id,
                display_label=instance.display_label,
                nports=instance.nports,
                position_x=instance.position_x,
                position_y=instance.position_y,
                block_kind=instance.block_kind,
                impedance_ohm=impedance_ohm,
                rotation_deg=instance.rotation_deg,
                mirror_horizontal=instance.mirror_horizontal,
                mirror_vertical=instance.mirror_vertical,
            )
            return

    def update_instance_transform(
        self,
        instance_id: str,
        *,
        rotation_deg: int,
        mirror_horizontal: bool,
        mirror_vertical: bool,
    ) -> None:
        for index, instance in enumerate(self.instances):
            if instance.instance_id != instance_id:
                continue
            self.instances[index] = CircuitBlockInstance(
                instance_id=instance.instance_id,
                source_file_id=instance.source_file_id,
                display_label=instance.display_label,
                nports=instance.nports,
                position_x=instance.position_x,
                position_y=instance.position_y,
                block_kind=instance.block_kind,
                impedance_ohm=instance.impedance_ohm,
                rotation_deg=rotation_deg,
                mirror_horizontal=mirror_horizontal,
                mirror_vertical=mirror_vertical,
            )
            return

    def remove_instance(self, instance_id: str) -> None:
        self.instances = [item for item in self.instances if item.instance_id != instance_id]
        self.connections = [
            item
            for item in self.connections
            if item.port_a.instance_id != instance_id and item.port_b.instance_id != instance_id
        ]
        self.rebuild_external_ports_from_instances()

    def add_connection(self, port_a: CircuitPortRef, port_b: CircuitPortRef) -> CircuitConnection:
        connection = CircuitConnection(
            connection_id=self.next_connection_id(),
            port_a=port_a,
            port_b=port_b,
        )
        self.connections.append(connection)
        return connection

    def remove_connection(self, connection_id: str) -> None:
        self.connections = [item for item in self.connections if item.connection_id != connection_id]

    def rebuild_external_ports_from_instances(self) -> None:
        external: List[ExternalPortAssignment] = []
        index = 1
        for instance in self.instances:
            if instance.block_kind == "port_ground":
                external.append(
                    ExternalPortAssignment(
                        external_port_number=index,
                        port_ref=CircuitPortRef(instance_id=instance.instance_id, port_number=1),
                    )
                )
                index += 1
                continue
            if instance.block_kind == "port_diff":
                for port_number in (1, 2):
                    external.append(
                        ExternalPortAssignment(
                            external_port_number=index,
                            port_ref=CircuitPortRef(instance_id=instance.instance_id, port_number=port_number),
                        )
                    )
                    index += 1
                continue
        self.external_ports = external

    def is_port_connected(self, port_ref: CircuitPortRef) -> bool:
        key = port_ref.key()
        return any(
            item.port_a.key() == key or item.port_b.key() == key for item in self.connections
        )

    def is_port_exported(self, port_ref: CircuitPortRef) -> bool:
        key = port_ref.key()
        return any(item.port_ref.key() == key for item in self.external_ports)

    def get_instance(self, instance_id: str) -> CircuitBlockInstance | None:
        for instance in self.instances:
            if instance.instance_id == instance_id:
                return instance
        return None

    def uses_file(self, file_id: str) -> bool:
        return any(
            item.block_kind == "touchstone" and item.source_file_id == file_id
            for item in self.instances
        )

    def validate(self) -> List[CircuitValidationIssue]:
        issues: List[CircuitValidationIssue] = []

        if self.sweep.fmin_hz <= 0 or self.sweep.fstep_hz <= 0:
            issues.append(CircuitValidationIssue("Sweep values must be greater than zero."))
        if self.sweep.fmax_hz < self.sweep.fmin_hz:
            issues.append(CircuitValidationIssue("Fmax must be greater than or equal to Fmin."))

        seen_external: Dict[int, Tuple[str, int]] = {}
        for item in self.external_ports:
            if item.external_port_number < 1:
                issues.append(CircuitValidationIssue("External port numbers must start from 1."))
            elif item.external_port_number in seen_external:
                issues.append(CircuitValidationIssue("External port numbers must be unique."))
            else:
                seen_external[item.external_port_number] = item.port_ref.key()

        for instance in self.instances:
            if instance.impedance_ohm <= 0:
                issues.append(
                    CircuitValidationIssue(
                        f"Impedance for {instance.display_label} must be greater than zero."
                    )
                )
        return issues

    def to_dict(self) -> dict:
        return {
            "instances": [item.to_dict() for item in self.instances],
            "connections": [item.to_dict() for item in self.connections],
            "external_ports": [item.to_dict() for item in self.external_ports],
            "sweep": self.sweep.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "CircuitDocument":
        doc = cls(
            instances=[
                CircuitBlockInstance.from_dict(item)
                for item in payload.get("instances", [])
                if isinstance(item, dict)
            ],
            connections=[
                CircuitConnection.from_dict(item)
                for item in payload.get("connections", [])
                if isinstance(item, dict)
            ],
            external_ports=[
                ExternalPortAssignment.from_dict(item)
                for item in payload.get("external_ports", [])
                if isinstance(item, dict)
            ],
            sweep=FrequencySweepSpec.from_dict(payload.get("sweep", {})),
        )
        # External ports are derived from dedicated port blocks.
        doc.rebuild_external_ports_from_instances()
        return doc