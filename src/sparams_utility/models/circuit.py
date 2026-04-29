from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

PRBS_CHOICES = ["PRBS-7", "PRBS-8", "PRBS-9", "PRBS-10", "PRBS-11", "PRBS-12", "PRBS-13", "PRBS-15", "PRBS-20", "PRBS-23", "PRBS-31"]
ENCODING_CHOICES = ["None", "8b10b", "64b66b", "128b130b", "PAM4"]
TRANSIENT_POLARITY_CHOICES = ["Positive", "Negative"]


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
    random_noise_v: float = 0.0  # σ of receiver random voltage noise (Vrn)
    source_impedance_ohm: float = 0.0  # Driver Thevenin source impedance (0 = ideal source, matches ADS ExcludeLoad=yes)
    maximal_length_lfsr: bool = False  # Generate exactly one full LFSR period (2^N − 1 bits)

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
            "random_noise_v": self.random_noise_v,
            "source_impedance_ohm": self.source_impedance_ohm,
            "maximal_length_lfsr": self.maximal_length_lfsr,
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
            random_noise_v=float(payload.get("random_noise_v", 0.0)),
            source_impedance_ohm=float(payload.get("source_impedance_ohm", 0.0)),
            maximal_length_lfsr=bool(payload.get("maximal_length_lfsr", False)),
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
class TransientSourceSpec:
    amplitude_v: float = 0.4
    polarity: str = "Positive"
    rise_time_s: float = 35e-12
    fall_time_s: float = 35e-12
    delay_s: float = 0.0
    pulse_width_s: float = 250e-12

    def to_dict(self) -> dict:
        return {
            "amplitude_v": self.amplitude_v,
            "polarity": self.polarity,
            "rise_time_s": self.rise_time_s,
            "fall_time_s": self.fall_time_s,
            "delay_s": self.delay_s,
            "pulse_width_s": self.pulse_width_s,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "TransientSourceSpec":
        polarity = str(payload.get("polarity", "Positive"))
        if polarity not in TRANSIENT_POLARITY_CHOICES:
            polarity = "Positive"
        return cls(
            amplitude_v=float(payload.get("amplitude_v", 0.4)),
            polarity=polarity,
            rise_time_s=float(payload.get("rise_time_s", 35e-12)),
            fall_time_s=float(payload.get("fall_time_s", 35e-12)),
            delay_s=float(payload.get("delay_s", 0.0)),
            pulse_width_s=float(payload.get("pulse_width_s", 250e-12)),
        )


@dataclass(frozen=True)
class SubstrateSpec:
    """Physical / electrical parameters of a PCB substrate stackup.

    Used by transmission-line blocks (microstrip, stripline, …) to compute
    Z0, εeff and per-unit-length loss. Stored on the substrate block
    instance itself so the schematic can host multiple stackups in parallel.
    """

    epsilon_r: float = 4.3            # relative dielectric constant
    loss_tangent: float = 0.02        # tan δ
    height_m: float = 200e-6          # h: dielectric thickness for microstrip (m).
                                      # For stripline this is derived from
                                      # h_top + t + h_bottom on save.
    conductor_thickness_m: float = 35e-6  # t: copper thickness (m)
    conductivity_s_per_m: float = 5.8e7   # σ: conductor conductivity (Cu)
    roughness_rq_m: float = 0.0       # surface roughness Rq (m), 0 = smooth
    # Stripline-only geometry: distances from the conductor surfaces to the
    # top/bottom ground planes (m). Ignored for microstrip substrates.
    stripline_h_top_m: float = 82.5e-6     # gap above conductor (m)
    stripline_h_bottom_m: float = 82.5e-6  # gap below conductor (m)

    def to_dict(self) -> dict:
        return {
            "epsilon_r": self.epsilon_r,
            "loss_tangent": self.loss_tangent,
            "height_m": self.height_m,
            "conductor_thickness_m": self.conductor_thickness_m,
            "conductivity_s_per_m": self.conductivity_s_per_m,
            "roughness_rq_m": self.roughness_rq_m,
            "stripline_h_top_m": self.stripline_h_top_m,
            "stripline_h_bottom_m": self.stripline_h_bottom_m,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "SubstrateSpec":
        h = float(payload.get("height_m", 200e-6))
        t = float(payload.get("conductor_thickness_m", 35e-6))
        # Backward compatibility: derive h_top/h_bottom from the legacy
        # `stripline_offset_m` field if the new fields are not present.
        if "stripline_h_top_m" in payload or "stripline_h_bottom_m" in payload:
            h_top = float(payload.get("stripline_h_top_m", max(0.0, (h - t) / 2.0)))
            h_bottom = float(payload.get("stripline_h_bottom_m", max(0.0, (h - t) / 2.0)))
        else:
            half = max(0.0, (h - t) / 2.0)
            offset = float(payload.get("stripline_offset_m", 0.0))
            # Legacy convention: positive offset = conductor toward top GND.
            h_top = max(0.0, half - offset)
            h_bottom = max(0.0, half + offset)
        return cls(
            epsilon_r=float(payload.get("epsilon_r", 4.3)),
            loss_tangent=float(payload.get("loss_tangent", 0.02)),
            height_m=h,
            conductor_thickness_m=t,
            conductivity_s_per_m=float(payload.get("conductivity_s_per_m", 5.8e7)),
            roughness_rq_m=float(payload.get("roughness_rq_m", 0.0)),
            stripline_h_top_m=h_top,
            stripline_h_bottom_m=h_bottom,
        )


@dataclass(frozen=True)
class TransmissionLineSpec:
    """Geometry of a transmission-line block.

    The block references a `SubstrateSpec` that lives on a separate
    substrate instance in the same schematic by its display label.
    The S-parameters of the line are synthesised on the fly from this
    spec + the referenced substrate at solve time and never persisted
    as a Touchstone file.
    """

    # microstrip | stripline | microstrip_coupled | stripline_coupled |
    # tline_cpw | tline_cpw_coupled | taper
    line_kind: str = "microstrip"
    substrate_name: str = ""        # display_label of the SubstrateSpec block to use
    width_m: float = 200e-6         # W: conductor width (m)
    length_m: float = 10e-3         # L: physical length (m)
    spacing_m: float = 200e-6       # S: edge-to-edge spacing (coupled only, m)
    z0_ref_ohm: float = 50.0        # Reference impedance for the synthesised S-matrix (per port)
    width_end_m: float = 0.0        # Taper end width (m); 0 = same as width_m
    taper_profile: str = "linear"   # linear | exponential | klopfenstein
    cpw_slot_m: float = 150e-6      # CPW slot width (m)

    def to_dict(self) -> dict:
        return {
            "line_kind": self.line_kind,
            "substrate_name": self.substrate_name,
            "width_m": self.width_m,
            "length_m": self.length_m,
            "spacing_m": self.spacing_m,
            "z0_ref_ohm": self.z0_ref_ohm,
            "width_end_m": self.width_end_m,
            "taper_profile": self.taper_profile,
            "cpw_slot_m": self.cpw_slot_m,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "TransmissionLineSpec":
        return cls(
            line_kind=str(payload.get("line_kind", "microstrip")),
            substrate_name=str(payload.get("substrate_name", "")),
            width_m=float(payload.get("width_m", 200e-6)),
            length_m=float(payload.get("length_m", 10e-3)),
            spacing_m=float(payload.get("spacing_m", 200e-6)),
            z0_ref_ohm=float(payload.get("z0_ref_ohm", 50.0)),
            width_end_m=float(payload.get("width_end_m", 0.0)),
            taper_profile=str(payload.get("taper_profile", "linear")),
            cpw_slot_m=float(payload.get("cpw_slot_m", 150e-6)),
        )


@dataclass(frozen=True)
class AttenuatorSpec:
    """Frequency-flat 2-port resistive attenuator."""

    attenuation_db: float = 6.0
    z0_ref_ohm: float = 50.0

    def to_dict(self) -> dict:
        return {
            "attenuation_db": self.attenuation_db,
            "z0_ref_ohm": self.z0_ref_ohm,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "AttenuatorSpec":
        return cls(
            attenuation_db=float(payload.get("attenuation_db", 6.0)),
            z0_ref_ohm=float(payload.get("z0_ref_ohm", 50.0)),
        )


@dataclass(frozen=True)
class CirculatorSpec:
    """Frequency-flat 3-port circulator."""

    insertion_loss_db: float = 0.3
    isolation_db: float = 30.0
    return_loss_db: float = 25.0
    direction: str = "cw"           # "cw" | "ccw"
    z0_ref_ohm: float = 50.0

    def to_dict(self) -> dict:
        return {
            "insertion_loss_db": self.insertion_loss_db,
            "isolation_db": self.isolation_db,
            "return_loss_db": self.return_loss_db,
            "direction": self.direction,
            "z0_ref_ohm": self.z0_ref_ohm,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "CirculatorSpec":
        direction = str(payload.get("direction", "cw"))
        if direction not in ("cw", "ccw"):
            direction = "cw"
        return cls(
            insertion_loss_db=float(payload.get("insertion_loss_db", 0.3)),
            isolation_db=float(payload.get("isolation_db", 30.0)),
            return_loss_db=float(payload.get("return_loss_db", 25.0)),
            direction=direction,
            z0_ref_ohm=float(payload.get("z0_ref_ohm", 50.0)),
        )


@dataclass(frozen=True)
class CouplerSpec:
    """Frequency-flat 4-port directional/branch-line coupler."""

    kind: str = "branch_line_90"
    coupling_db: float = 3.0
    insertion_loss_db: float = 0.3
    isolation_db: float = 30.0
    return_loss_db: float = 25.0
    z0_ref_ohm: float = 50.0

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "coupling_db": self.coupling_db,
            "insertion_loss_db": self.insertion_loss_db,
            "isolation_db": self.isolation_db,
            "return_loss_db": self.return_loss_db,
            "z0_ref_ohm": self.z0_ref_ohm,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "CouplerSpec":
        return cls(
            kind=str(payload.get("kind", "branch_line_90")),
            coupling_db=float(payload.get("coupling_db", 3.0)),
            insertion_loss_db=float(payload.get("insertion_loss_db", 0.3)),
            isolation_db=float(payload.get("isolation_db", 30.0)),
            return_loss_db=float(payload.get("return_loss_db", 25.0)),
            z0_ref_ohm=float(payload.get("z0_ref_ohm", 50.0)),
        )


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
    transient_source_spec: Optional[TransientSourceSpec] = None
    substrate_spec: Optional[SubstrateSpec] = None
    transmission_line_spec: Optional[TransmissionLineSpec] = None
    attenuator_spec: Optional[AttenuatorSpec] = None
    circulator_spec: Optional[CirculatorSpec] = None
    coupler_spec: Optional[CouplerSpec] = None

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
        if self.transient_source_spec is not None:
            d["transient_source_spec"] = self.transient_source_spec.to_dict()
        if self.substrate_spec is not None:
            d["substrate_spec"] = self.substrate_spec.to_dict()
        if self.transmission_line_spec is not None:
            d["transmission_line_spec"] = self.transmission_line_spec.to_dict()
        if self.attenuator_spec is not None:
            d["attenuator_spec"] = self.attenuator_spec.to_dict()
        if self.circulator_spec is not None:
            d["circulator_spec"] = self.circulator_spec.to_dict()
        if self.coupler_spec is not None:
            d["coupler_spec"] = self.coupler_spec.to_dict()
        return d

    @classmethod
    def from_dict(cls, payload: dict) -> "CircuitBlockInstance":
        ds = payload.get("driver_spec")
        transient_spec = payload.get("transient_source_spec")
        substrate_spec = payload.get("substrate_spec")
        tline_spec = payload.get("transmission_line_spec")
        att_spec = payload.get("attenuator_spec")
        circ_spec = payload.get("circulator_spec")
        coup_spec = payload.get("coupler_spec")
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
            transient_source_spec=(
                TransientSourceSpec.from_dict(transient_spec)
                if isinstance(transient_spec, dict)
                else None
            ),
            substrate_spec=(
                SubstrateSpec.from_dict(substrate_spec)
                if isinstance(substrate_spec, dict)
                else None
            ),
            transmission_line_spec=(
                TransmissionLineSpec.from_dict(tline_spec)
                if isinstance(tline_spec, dict)
                else None
            ),
            attenuator_spec=(
                AttenuatorSpec.from_dict(att_spec)
                if isinstance(att_spec, dict)
                else None
            ),
            circulator_spec=(
                CirculatorSpec.from_dict(circ_spec)
                if isinstance(circ_spec, dict)
                else None
            ),
            coupler_spec=(
                CouplerSpec.from_dict(coup_spec)
                if isinstance(coup_spec, dict)
                else None
            ),
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
        transient_source_spec: Optional[TransientSourceSpec] = None,
        substrate_spec: Optional[SubstrateSpec] = None,
        transmission_line_spec: Optional[TransmissionLineSpec] = None,
        attenuator_spec: Optional[AttenuatorSpec] = None,
        circulator_spec: Optional[CirculatorSpec] = None,
        coupler_spec: Optional[CouplerSpec] = None,
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
            transient_source_spec=transient_source_spec,
            substrate_spec=substrate_spec,
            transmission_line_spec=transmission_line_spec,
            attenuator_spec=attenuator_spec,
            circulator_spec=circulator_spec,
            coupler_spec=coupler_spec,
        )
        self.instances.append(instance)
        self.rebuild_external_ports_from_instances()
        return instance

    def update_instance_position(self, instance_id: str, x: float, y: float) -> None:
        for index, instance in enumerate(self.instances):
            if instance.instance_id != instance_id:
                continue
            self.instances[index] = replace(instance, position_x=x, position_y=y)
            return

    def update_instance_impedance(self, instance_id: str, impedance_ohm: float) -> None:
        for index, instance in enumerate(self.instances):
            if instance.instance_id != instance_id:
                continue
            self.instances[index] = replace(instance, impedance_ohm=impedance_ohm)
            return

    def update_instance_display_label(self, instance_id: str, display_label: str) -> None:
        for index, instance in enumerate(self.instances):
            if instance.instance_id != instance_id:
                continue
            self.instances[index] = replace(instance, display_label=display_label)
            return

    def update_instance_symbol_scale(self, instance_id: str, symbol_scale: float) -> None:
        for index, instance in enumerate(self.instances):
            if instance.instance_id != instance_id:
                continue
            self.instances[index] = replace(
                instance,
                symbol_scale=max(0.5, min(3.0, symbol_scale)),
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
            self.instances[index] = replace(
                instance,
                rotation_deg=rotation_deg,
                mirror_horizontal=mirror_horizontal,
                mirror_vertical=mirror_vertical,
            )
            return

    def update_instance_driver_spec(self, instance_id: str, driver_spec: Optional[DriverSpec]) -> None:
        for index, instance in enumerate(self.instances):
            if instance.instance_id != instance_id:
                continue
            self.instances[index] = replace(instance, driver_spec=driver_spec)
            return

    def update_instance_transient_source_spec(
        self,
        instance_id: str,
        transient_source_spec: Optional[TransientSourceSpec],
    ) -> None:
        for index, instance in enumerate(self.instances):
            if instance.instance_id != instance_id:
                continue
            self.instances[index] = replace(instance, transient_source_spec=transient_source_spec)
            return

    def update_instance_substrate_spec(
        self,
        instance_id: str,
        substrate_spec: Optional[SubstrateSpec],
    ) -> None:
        for index, instance in enumerate(self.instances):
            if instance.instance_id != instance_id:
                continue
            self.instances[index] = replace(instance, substrate_spec=substrate_spec)
            return

    def update_instance_transmission_line_spec(
        self,
        instance_id: str,
        transmission_line_spec: Optional[TransmissionLineSpec],
    ) -> None:
        for index, instance in enumerate(self.instances):
            if instance.instance_id != instance_id:
                continue
            self.instances[index] = replace(
                instance, transmission_line_spec=transmission_line_spec
            )
            return

    def update_instance_attenuator_spec(
        self,
        instance_id: str,
        attenuator_spec: Optional[AttenuatorSpec],
    ) -> None:
        for index, instance in enumerate(self.instances):
            if instance.instance_id != instance_id:
                continue
            self.instances[index] = replace(instance, attenuator_spec=attenuator_spec)
            return

    def update_instance_circulator_spec(
        self,
        instance_id: str,
        circulator_spec: Optional[CirculatorSpec],
    ) -> None:
        for index, instance in enumerate(self.instances):
            if instance.instance_id != instance_id:
                continue
            self.instances[index] = replace(instance, circulator_spec=circulator_spec)
            return

    def update_instance_coupler_spec(
        self,
        instance_id: str,
        coupler_spec: Optional[CouplerSpec],
    ) -> None:
        for index, instance in enumerate(self.instances):
            if instance.instance_id != instance_id:
                continue
            self.instances[index] = replace(instance, coupler_spec=coupler_spec)
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
            if instance.block_kind in {"transient_step_se", "transient_pulse_se"}:
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
            if instance.block_kind == "scope_se":
                external.append(
                    ExternalPortAssignment(
                        external_port_number=index,
                        port_ref=CircuitPortRef(instance_id=instance.instance_id, port_number=1),
                    )
                )
                index += 1
                continue
            if instance.block_kind == "scope_diff":
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