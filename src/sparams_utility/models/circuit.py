from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

PRBS_CHOICES = ["PRBS-7", "PRBS-8", "PRBS-9", "PRBS-10", "PRBS-11", "PRBS-12", "PRBS-13", "PRBS-15", "PRBS-20", "PRBS-23", "PRBS-31"]
ENCODING_CHOICES = ["None", "8b10b", "64b66b", "128b130b", "PAM4"]


@dataclass(frozen=True)
class DriverSpec:
    voltage_high_v: float = 0.4
    voltage_low_v: float = 0.0
    rise_time_s: float = 35e-12
    fall_time_s: float = 35e-12
    bitrate_gbps: float = 10.0
    prbs_pattern: str = "PRBS-7"
    encoding: str = "None"
    num_bits: int = 2**13

    def to_dict(self) -> dict:
        return {
            "voltage_high_v": self.voltage_high_v,
            "voltage_low_v": self.voltage_low_v,
            "rise_time_s": self.rise_time_s,
            "fall_time_s": self.fall_time_s,
            "bitrate_gbps": self.bitrate_gbps,
            "prbs_pattern": self.prbs_pattern,
            "encoding": self.encoding,
            "num_bits": self.num_bits,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "DriverSpec":
        return cls(
            voltage_high_v=float(payload.get("voltage_high_v", 0.45)),
            voltage_low_v=float(payload.get("voltage_low_v", -0.45)),
            rise_time_s=float(payload.get("rise_time_s", 25e-12)),
            fall_time_s=float(payload.get("fall_time_s", 25e-12)),
            bitrate_gbps=float(payload.get("bitrate_gbps", 10.0)),
            prbs_pattern=str(payload.get("prbs_pattern", "PRBS-7")),
            encoding=str(payload.get("encoding", "None")),
            num_bits=int(payload.get("num_bits", 2**13)),
        )


@dataclass(frozen=True)
class ChannelSimSpec:
    num_bits: int = 2**13

    def to_dict(self) -> dict:
        return {"num_bits": self.num_bits}

    @classmethod
    def from_dict(cls, payload: dict) -> "ChannelSimSpec":
        return cls(num_bits=int(payload.get("num_bits", 2**13)))


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
    symbol_scale: float = 1.0
    rotation_deg: int = 0
    mirror_horizontal: bool = False
    mirror_vertical: bool = False
    driver_spec: Optional[DriverSpec] = None

    def to_dict(self) -> dict:
        d = {
            "instance_id": self.instance_id,
            "source_file_id": self.source_file_id,
            "display_label": self.display_label,
            "nports": self.nports,
            "position_x": self.position_x,
            "position_y": self.position_y,
            "block_kind": self.block_kind,
            "impedance_ohm": self.impedance_ohm,
            "symbol_scale": self.symbol_scale,
            "rotation_deg": self.rotation_deg,
            "mirror_horizontal": self.mirror_horizontal,
            "mirror_vertical": self.mirror_vertical,
        }
        if self.driver_spec is not None:
            d["driver_spec"] = self.driver_spec.to_dict()
        return d

    @classmethod
    def from_dict(cls, payload: dict) -> "CircuitBlockInstance":
        ds = payload.get("driver_spec")
        return cls(
            instance_id=str(payload.get("instance_id", "")),
            source_file_id=str(payload.get("source_file_id", "")),
            display_label=str(payload.get("display_label", "Block")),
            nports=int(payload.get("nports", 0)),
            position_x=float(payload.get("position_x", 0.0)),
            position_y=float(payload.get("position_y", 0.0)),
            block_kind=str(payload.get("block_kind", "touchstone")),
            impedance_ohm=float(payload.get("impedance_ohm", 50.0)),
            symbol_scale=float(payload.get("symbol_scale", 1.0)),
            rotation_deg=int(payload.get("rotation_deg", 0)),
            mirror_horizontal=bool(payload.get("mirror_horizontal", False)),
            mirror_vertical=bool(payload.get("mirror_vertical", False)),
            driver_spec=DriverSpec.from_dict(ds) if isinstance(ds, dict) else None,
        )


@dataclass(frozen=True)
class CircuitConnection:
    connection_id: str
    port_a: CircuitPortRef
    port_b: CircuitPortRef
    # Optional list of (x, y) waypoints that control the wire path.
    # Each waypoint is a grid-snapped scene coordinate stored as a tuple.
    waypoints: Tuple[Tuple[float, float], ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        d: dict = {
            "connection_id": self.connection_id,
            "port_a": self.port_a.to_dict(),
            "port_b": self.port_b.to_dict(),
        }
        if self.waypoints:
            d["waypoints"] = [list(wp) for wp in self.waypoints]
        return d

    @classmethod
    def from_dict(cls, payload: dict) -> "CircuitConnection":
        raw_wp = payload.get("waypoints", [])
        waypoints: Tuple[Tuple[float, float], ...] = tuple(
            (float(w[0]), float(w[1])) for w in raw_wp if len(w) >= 2
        )
        return cls(
            connection_id=str(payload.get("connection_id", "")),
            port_a=CircuitPortRef.from_dict(payload.get("port_a", {})),
            port_b=CircuitPortRef.from_dict(payload.get("port_b", {})),
            waypoints=waypoints,
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
class DifferentialPortAssignment:
    """Maps one external differential port to its + and − single-ended nodes."""

    external_port_number: int
    port_ref_plus: CircuitPortRef   # port 1 of port_diff block
    port_ref_minus: CircuitPortRef  # port 2 of port_diff block

    def to_dict(self) -> dict:
        return {
            "external_port_number": self.external_port_number,
            "port_ref_plus": self.port_ref_plus.to_dict(),
            "port_ref_minus": self.port_ref_minus.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "DifferentialPortAssignment":
        return cls(
            external_port_number=int(payload.get("external_port_number", 0)),
            port_ref_plus=CircuitPortRef.from_dict(payload.get("port_ref_plus", {})),
            port_ref_minus=CircuitPortRef.from_dict(payload.get("port_ref_minus", {})),
        )


@dataclass(frozen=True)
class CircuitValidationIssue:
    message: str


@dataclass
class CircuitDocument:
    instances: List[CircuitBlockInstance] = field(default_factory=list)
    connections: List[CircuitConnection] = field(default_factory=list)
    external_ports: List[ExternalPortAssignment] = field(default_factory=list)
    differential_ports: List[DifferentialPortAssignment] = field(default_factory=list)
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
        symbol_scale: float = 1.0,
        rotation_deg: int = 0,
        mirror_horizontal: bool = False,
        mirror_vertical: bool = False,
        driver_spec: Optional[DriverSpec] = None,
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
            symbol_scale=symbol_scale,
            rotation_deg=rotation_deg,
            mirror_horizontal=mirror_horizontal,
            mirror_vertical=mirror_vertical,
            driver_spec=driver_spec,
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
                symbol_scale=instance.symbol_scale,
                rotation_deg=instance.rotation_deg,
                mirror_horizontal=instance.mirror_horizontal,
                mirror_vertical=instance.mirror_vertical,
                driver_spec=instance.driver_spec,
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
                symbol_scale=instance.symbol_scale,
                rotation_deg=instance.rotation_deg,
                mirror_horizontal=instance.mirror_horizontal,
                mirror_vertical=instance.mirror_vertical,
                driver_spec=instance.driver_spec,
            )
            return

    def update_instance_symbol_scale(self, instance_id: str, symbol_scale: float) -> None:
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
                symbol_scale=max(0.5, min(3.0, symbol_scale)),
                rotation_deg=instance.rotation_deg,
                mirror_horizontal=instance.mirror_horizontal,
                mirror_vertical=instance.mirror_vertical,
                driver_spec=instance.driver_spec,
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
                symbol_scale=instance.symbol_scale,
                rotation_deg=rotation_deg,
                mirror_horizontal=mirror_horizontal,
                mirror_vertical=mirror_vertical,
                driver_spec=instance.driver_spec,
            )
            return

    def update_instance_driver_spec(self, instance_id: str, driver_spec: Optional[DriverSpec]) -> None:
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
                symbol_scale=instance.symbol_scale,
                rotation_deg=instance.rotation_deg,
                mirror_horizontal=instance.mirror_horizontal,
                mirror_vertical=instance.mirror_vertical,
                driver_spec=driver_spec,
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

    def add_connection(
        self,
        port_a: CircuitPortRef,
        port_b: CircuitPortRef,
        waypoints: Tuple[Tuple[float, float], ...] = (),
    ) -> CircuitConnection:
        connection = CircuitConnection(
            connection_id=self.next_connection_id(),
            port_a=port_a,
            port_b=port_b,
            waypoints=waypoints,
        )
        self.connections.append(connection)
        return connection

    def remove_connection(self, connection_id: str) -> None:
        self.connections = [item for item in self.connections if item.connection_id != connection_id]

    def update_connection_waypoints(
        self, connection_id: str, waypoints: Tuple[Tuple[float, float], ...]
    ) -> None:
        self.connections = [
            CircuitConnection(
                connection_id=c.connection_id,
                port_a=c.port_a,
                port_b=c.port_b,
                waypoints=waypoints,
            )
            if c.connection_id == connection_id
            else c
            for c in self.connections
        ]

    def rebuild_external_ports_from_instances(self) -> None:
        external: List[ExternalPortAssignment] = []
        differential: List[DifferentialPortAssignment] = []
        index = 1
        for instance in self.instances:
            if instance.block_kind in {"gnd", "net_node"}:
                continue
            if instance.block_kind == "port_ground":
                external.append(
                    ExternalPortAssignment(
                        external_port_number=index,
                        port_ref=CircuitPortRef(instance_id=instance.instance_id, port_number=1),
                    )
                )
                index += 1
                continue
            if instance.block_kind == "driver_se":
                external.append(
                    ExternalPortAssignment(
                        external_port_number=index,
                        port_ref=CircuitPortRef(instance_id=instance.instance_id, port_number=1),
                    )
                )
                index += 1
                continue
            if instance.block_kind == "driver_diff":
                differential.append(
                    DifferentialPortAssignment(
                        external_port_number=index,
                        port_ref_plus=CircuitPortRef(instance_id=instance.instance_id, port_number=1),
                        port_ref_minus=CircuitPortRef(instance_id=instance.instance_id, port_number=2),
                    )
                )
                index += 1
                continue
            if instance.block_kind == "port_diff":
                differential.append(
                    DifferentialPortAssignment(
                        external_port_number=index,
                        port_ref_plus=CircuitPortRef(instance_id=instance.instance_id, port_number=1),
                        port_ref_minus=CircuitPortRef(instance_id=instance.instance_id, port_number=2),
                    )
                )
                index += 1
                continue
            if instance.block_kind == "eyescope_se":
                external.append(
                    ExternalPortAssignment(
                        external_port_number=index,
                        port_ref=CircuitPortRef(instance_id=instance.instance_id, port_number=1),
                    )
                )
                index += 1
                continue
            if instance.block_kind == "eyescope_diff":
                differential.append(
                    DifferentialPortAssignment(
                        external_port_number=index,
                        port_ref_plus=CircuitPortRef(instance_id=instance.instance_id, port_number=1),
                        port_ref_minus=CircuitPortRef(instance_id=instance.instance_id, port_number=2),
                    )
                )
                index += 1
                continue
        self.external_ports = external
        self.differential_ports = differential

    def is_port_connected(self, port_ref: CircuitPortRef) -> bool:
        key = port_ref.key()
        return any(
            item.port_a.key() == key or item.port_b.key() == key for item in self.connections
        )

    def is_port_exported(self, port_ref: CircuitPortRef) -> bool:
        key = port_ref.key()
        if any(item.port_ref.key() == key for item in self.external_ports):
            return True
        return any(
            item.port_ref_plus.key() == key or item.port_ref_minus.key() == key
            for item in self.differential_ports
        )

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

        for item in self.differential_ports:
            if item.external_port_number < 1:
                issues.append(CircuitValidationIssue("External port numbers must start from 1."))
            elif item.external_port_number in seen_external:
                issues.append(CircuitValidationIssue("External port numbers must be unique."))
            else:
                seen_external[item.external_port_number] = item.port_ref_plus.key()

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
            "differential_ports": [item.to_dict() for item in self.differential_ports],
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
            differential_ports=[
                DifferentialPortAssignment.from_dict(item)
                for item in payload.get("differential_ports", [])
                if isinstance(item, dict)
            ],
            sweep=FrequencySweepSpec.from_dict(payload.get("sweep", {})),
        )
        # External ports are derived from dedicated port blocks.
        doc.rebuild_external_ports_from_instances()
        return doc