# params/ppe_mapping.py

from dataclasses import dataclass
from params.structure import WorkEnvironment, HazardFactor, PPEItem

@dataclass(frozen=True)
class HazardPPE:
    hazard: HazardFactor
    ppe: list[PPEItem]

# WorkEnvironment별 HazardPPE 리스트(= hazard -> ppe) 구조
WORK_ENV_TO_HAZARD_PPE: dict[WorkEnvironment, list[HazardPPE]] = {
    # -------------------------
    # W01: Cutting/Grinding
    # -------------------------
    WorkEnvironment.CUTTING_GRINDING_MACHINING_POLISHING: [
        HazardPPE(hazard=HazardFactor.CUTTING, ppe=[PPEItem.SAFETY_GOGGLES, PPEItem.SAFETY_SHOES, PPEItem.CUT_RESISTANT_GLOVES, PPEItem.FACE_SHIELD]),
        HazardPPE(hazard=HazardFactor.IMPACT, ppe=[PPEItem.A_TYPE_SAFETY_HELMET, PPEItem.SAFETY_HELMET, PPEItem.SAFETY_SHOES]),
        HazardPPE(hazard=HazardFactor.NOISE, ppe=[PPEItem.HEARING_PROTECTION]),
        HazardPPE(hazard=HazardFactor.VIBRATION, ppe=[PPEItem.ANTI_VIBRATION_GLOVES]),
        HazardPPE(hazard=HazardFactor.DUST, ppe=[PPEItem.DUST_MASK]),
        # [New] 기계 얽힘 추가
        HazardPPE(hazard=HazardFactor.ENTANGLEMENT_CRUSHING_FROM_ROTATING_MACHINERY, ppe=[PPEItem.ANTI_PINCH_GLOVES, PPEItem.SAFETY_SHOES, PPEItem.PROTECTIVE_CLOTHING]),
    ],
    # -------------------------
    # W02: Drilling/Demolition
    # -------------------------
    WorkEnvironment.DRILLING_CRUSHING_DEMOLITION_DISMANTLING: [
        HazardPPE(hazard=HazardFactor.DUST, ppe=[PPEItem.DUST_MASK, PPEItem.SAFETY_GOGGLES]),
        HazardPPE(hazard=HazardFactor.NOISE, ppe=[PPEItem.HEARING_PROTECTION]),
        HazardPPE(hazard=HazardFactor.VIBRATION, ppe=[PPEItem.ANTI_VIBRATION_GLOVES]),
        HazardPPE(hazard=HazardFactor.IMPACT, ppe=[PPEItem.A_TYPE_SAFETY_HELMET, PPEItem.SAFETY_HELMET, PPEItem.SAFETY_SHOES]),
        # [New] 낙하물 위험 추가
        HazardPPE(hazard=HazardFactor.FALLING_OBJECTS, ppe=[PPEItem.A_TYPE_SAFETY_HELMET, PPEItem.SAFETY_HELMET, PPEItem.SAFETY_SHOES]),
    ],
    # -------------------------
    # W03: Welding/Hot Work
    # -------------------------
    WorkEnvironment.WELDING_CUTTING_HOT_WORK: [
        HazardPPE(hazard=HazardFactor.LIGHTING, ppe=[PPEItem.WELDING_MASK, PPEItem.FACE_SHIELD_OR_WELDING_GOGGLES]),
        HazardPPE(hazard=HazardFactor.HIGH_TEMPERATURE, ppe=[PPEItem.HEAT_RESISTANT_MASK, PPEItem.HEAT_RESISTANT_GLOVES, PPEItem.HEAT_RESISTANT_CLOTHING, PPEItem.HEAT_RESISTANT_SHOES]),
        HazardPPE(hazard=HazardFactor.IMPACT, ppe=[PPEItem.FACE_SHIELD, PPEItem.A_TYPE_SAFETY_HELMET, PPEItem.SAFETY_SHOES]),
        HazardPPE(hazard=HazardFactor.CONFINED_SPACE, ppe=[PPEItem.AIR_SUPPLIED_RESPIRATOR]),
        # [New] 핫스파크 및 가스/에어로졸 추가
        HazardPPE(hazard=HazardFactor.HOT_SPARKS, ppe=[PPEItem.WELDING_MASK, PPEItem.HEAT_RESISTANT_CLOTHING, PPEItem.HEAT_RESISTANT_GLOVES]),
        HazardPPE(hazard=HazardFactor.CHEMICAL_AEROSOL, ppe=[PPEItem.GAS_RESPIRATOR_MASK, PPEItem.RESPIRATOR_CARTRIDGE_FILTER]),
    ],
    # -------------------------
    # W04: Chemical Handling
    # -------------------------
    WorkEnvironment.CHEMICAL_HANDLING_MIXING_PAINTING_CLEANING_COATING: [
        HazardPPE(hazard=HazardFactor.TOXIC_SUBSTANCE, ppe=[PPEItem.GAS_RESPIRATOR_MASK, PPEItem.IMPERMEABLE_PROTECTIVE_SUIT, PPEItem.CHEMICAL_RESISTANT_GLOVES, PPEItem.CHEMICAL_RESISTANT_SAFETY_BOOTS, PPEItem.RESPIRATOR_CARTRIDGE_FILTER]),
        HazardPPE(hazard=HazardFactor.CONFINED_SPACE, ppe=[PPEItem.AIR_SUPPLIED_RESPIRATOR]),
        # [New] 구체화된 화학 위험 추가
        HazardPPE(hazard=HazardFactor.CHEMICAL_EXPOSURE_HAZARD, ppe=[PPEItem.CHEMICAL_RESISTANT_GLOVES, PPEItem.IMPERMEABLE_PROTECTIVE_SUIT, PPEItem.SAFETY_GOGGLES]),
        HazardPPE(hazard=HazardFactor.CHEMICAL_AEROSOL, ppe=[PPEItem.GAS_RESPIRATOR_MASK, PPEItem.SAFETY_GOGGLES]),
    ],
    # -------------------------
    # W05: Powder/Cement/Wood
    # -------------------------
    WorkEnvironment.POWDER_CEMENT_WOODWORKING_ABRASIVE_DUST: [
        HazardPPE(hazard=HazardFactor.DUST, ppe=[PPEItem.DUST_MASK, PPEItem.SAFETY_GOGGLES]),
        HazardPPE(hazard=HazardFactor.IMPACT, ppe=[PPEItem.A_TYPE_SAFETY_HELMET, PPEItem.SAFETY_SHOES]),
    ],
    # -------------------------
    # W06: Electrical
    # -------------------------
    WorkEnvironment.ELECTRICAL_INSTALLATION_INSPECTION_MAINTENANCE: [
        HazardPPE(hazard=HazardFactor.ELECTRIC_SHOCK, ppe=[PPEItem.INSULATING_HELMET, PPEItem.INSULATED_GLOVES, PPEItem.INSULATING_BOOTS, PPEItem.ABE_TYPE_SAFETY_HELMET, PPEItem.DIELECTRIC_TOOLS]),
        HazardPPE(hazard=HazardFactor.CAUGHT_IN_OR_BETWEEN, ppe=[PPEItem.SAFETY_SHOES, PPEItem.PROTECTIVE_CLOTHING]),
        # [New] 고소 작업 병행 가능성
        HazardPPE(hazard=HazardFactor.FALL, ppe=[PPEItem.SAFETY_HARNESS, PPEItem.ABE_TYPE_SAFETY_HELMET]),
    ],
    # -------------------------
    # W07: Working at Height
    # -------------------------
    WorkEnvironment.WORKING_AT_HEIGHT_SCAFFOLD_LADDER_OPENINGS_TEMPORARY_WALKWAYS: [
        HazardPPE(hazard=HazardFactor.FALL, ppe=[PPEItem.SAFETY_HARNESS, PPEItem.AB_TYPE_SAFETY_HELMET, PPEItem.SAFETY_HELMET]),
        HazardPPE(hazard=HazardFactor.IMPACT, ppe=[PPEItem.A_TYPE_SAFETY_HELMET, PPEItem.SAFETY_SHOES]),
        HazardPPE(hazard=HazardFactor.SLIP, ppe=[PPEItem.SAFETY_SHOES, PPEItem.SAFETY_FOOTWEAR]),
        # [New] 낙하물 추가
        HazardPPE(hazard=HazardFactor.FALLING_OBJECTS, ppe=[PPEItem.AB_TYPE_SAFETY_HELMET, PPEItem.SAFETY_HELMET, PPEItem.SAFETY_SHOES]),
    ],
    # -------------------------
    # W08: Lifting/Crane
    # -------------------------
    WorkEnvironment.LIFTING_LOADING_UNLOADING_CRANE: [
        HazardPPE(hazard=HazardFactor.IMPACT, ppe=[PPEItem.A_TYPE_SAFETY_HELMET, PPEItem.SAFETY_SHOES]),
        HazardPPE(hazard=HazardFactor.CAUGHT_IN_OR_BETWEEN, ppe=[PPEItem.ANTI_PINCH_GLOVES, PPEItem.SAFETY_SHOES]),
        HazardPPE(hazard=HazardFactor.FALLING_OBJECTS, ppe=[PPEItem.A_TYPE_SAFETY_HELMET, PPEItem.SAFETY_SHOES]),
    ],
    # -------------------------
    # W09: Mobile Equipment
    # -------------------------
    WorkEnvironment.WORKING_NEAR_MOBILE_EQUIPMENT: [
        HazardPPE(hazard=HazardFactor.CAUGHT_IN_OR_BETWEEN, ppe=[PPEItem.ANTI_PINCH_GLOVES, PPEItem.SAFETY_SHOES, PPEItem.PROTECTIVE_CLOTHING]),
        HazardPPE(hazard=HazardFactor.IMPACT, ppe=[PPEItem.A_TYPE_SAFETY_HELMET, PPEItem.SAFETY_SHOES]),
        HazardPPE(hazard=HazardFactor.SLIP, ppe=[PPEItem.SAFETY_SHOES]),
    ],
    # -------------------------
    # W10: Rotating Machinery
    # -------------------------
    WorkEnvironment.ROTATING_MACHINERY_PRESS_CONVEYOR_EQUIPMENT: [
        HazardPPE(hazard=HazardFactor.CAUGHT_IN_OR_BETWEEN, ppe=[PPEItem.ANTI_PINCH_GLOVES, PPEItem.SAFETY_SHOES, PPEItem.PROTECTIVE_CLOTHING]),
        HazardPPE(hazard=HazardFactor.ENTANGLEMENT_CRUSHING_FROM_ROTATING_MACHINERY, ppe=[PPEItem.ANTI_PINCH_GLOVES, PPEItem.PROTECTIVE_CLOTHING]),
        HazardPPE(hazard=HazardFactor.CUTTING, ppe=[PPEItem.CUT_RESISTANT_GLOVES, PPEItem.SAFETY_GOGGLES]),
    ],
    # -------------------------
    # W11: Confined Space
    # -------------------------
    WorkEnvironment.CONFINED_SPACE_WORK: [
        HazardPPE(hazard=HazardFactor.CONFINED_SPACE, ppe=[PPEItem.AIR_SUPPLIED_RESPIRATOR, PPEItem.PROTECTIVE_CLOTHING]),
        HazardPPE(hazard=HazardFactor.FALL, ppe=[PPEItem.SAFETY_HARNESS, PPEItem.AIR_SUPPLIED_RESPIRATOR]),
        HazardPPE(hazard=HazardFactor.TOXIC_SUBSTANCE, ppe=[PPEItem.AIR_SUPPLIED_RESPIRATOR, PPEItem.GAS_RESPIRATOR_MASK]),
        HazardPPE(hazard=HazardFactor.BIOLOGICAL_INFECTIOUS, ppe=[PPEItem.IMPERMEABLE_PROTECTIVE_SUIT, PPEItem.AIR_SUPPLIED_RESPIRATOR]),
    ],
    # -------------------------
    # W12: Cold Work
    # -------------------------
    WorkEnvironment.LOW_TEMPERATURE_COLD_WORK: [
        HazardPPE(hazard=HazardFactor.LOW_TEMPERATURE, ppe=[PPEItem.COLD_PROTECTION_HOOD, PPEItem.COLD_PROTECTION_CLOTHING, PPEItem.COLD_PROTECTION_BOOTS, PPEItem.COLD_PROTECTION_GLOVES, PPEItem.INSULATING_COLD_BOOTS]),
        HazardPPE(hazard=HazardFactor.SLIP, ppe=[PPEItem.SAFETY_SHOES, PPEItem.SAFETY_FOOTWEAR]),
        # [New] 결빙 조건 추가
        HazardPPE(hazard=HazardFactor.ICE_CONDITIONS, ppe=[PPEItem.SAFETY_SHOES, PPEItem.SAFETY_FOOTWEAR]),
    ],
    # -------------------------
    # [New] W13: Healthcare / Biological
    # -------------------------
    WorkEnvironment.HEALTHCARE_PATIENT_CARE: [
        HazardPPE(hazard=HazardFactor.BIOLOGICAL_INFECTIOUS, ppe=[PPEItem.MEDICAL_GLOVES, PPEItem.PROTECTIVE_CLOTHING, PPEItem.FACE_SHIELD, PPEItem.DUST_MASK]), # 마스크는 상황에 따라 N95 등 매핑 가능
        HazardPPE(hazard=HazardFactor.CHEMICAL_EXPOSURE_HAZARD, ppe=[PPEItem.MEDICAL_GLOVES, PPEItem.SAFETY_GOGGLES]),
    ],
    # -------------------------
    # [New] W14: Tunnel / Underground
    # -------------------------
    WorkEnvironment.TUNNEL_UNDERGROUND_WORK: [
        HazardPPE(hazard=HazardFactor.FALLING_OBJECTS, ppe=[PPEItem.A_TYPE_SAFETY_HELMET, PPEItem.SAFETY_HELMET, PPEItem.SAFETY_SHOES]),
        HazardPPE(hazard=HazardFactor.DUST, ppe=[PPEItem.DUST_MASK]),
        HazardPPE(hazard=HazardFactor.CONFINED_SPACE, ppe=[PPEItem.AIR_SUPPLIED_RESPIRATOR]),
        HazardPPE(hazard=HazardFactor.SLIP, ppe=[PPEItem.SAFETY_SHOES]),
        HazardPPE(hazard=HazardFactor.LIGHTING, ppe=[PPEItem.SAFETY_HELMET, PPEItem.PROTECTIVE_CLOTHING]), # 시인성 확보
    ],
    # -------------------------
    # [New] W15: Rooftop Solar
    # -------------------------
    WorkEnvironment.ROOFTOP_SOLAR_INSTALLATION: [
        HazardPPE(hazard=HazardFactor.FALL, ppe=[PPEItem.SAFETY_HARNESS, PPEItem.AB_TYPE_SAFETY_HELMET, PPEItem.SAFETY_HELMET]),
        HazardPPE(hazard=HazardFactor.ELECTRIC_SHOCK, ppe=[PPEItem.INSULATED_GLOVES, PPEItem.INSULATING_BOOTS, PPEItem.DIELECTRIC_TOOLS]),
        HazardPPE(hazard=HazardFactor.SLIP, ppe=[PPEItem.SAFETY_SHOES]),
        HazardPPE(hazard=HazardFactor.HIGH_TEMPERATURE, ppe=[PPEItem.PROTECTIVE_CLOTHING, PPEItem.SAFETY_GOGGLES]), # 야외 작업
    ],
}

def get_required_ppe(
    work_env: WorkEnvironment, hazards: list[HazardFactor]
) -> list[PPEItem]:
    """
    특정 작업환경에서 hazards(복수) 조건일 때의 PPE 합집합을 반환.
    """
    mapping = WORK_ENV_TO_HAZARD_PPE.get(work_env, [])
    ppe_set = []
    for hp in mapping:
        if hp.hazard in hazards:
            for item in hp.ppe:
                if item not in ppe_set:
                    ppe_set.append(item)
    return ppe_set
