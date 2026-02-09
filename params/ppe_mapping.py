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
        HazardPPE(hazard=HazardFactor.IMPACT, ppe=[PPEItem.AB_TYPE_SAFETY_HELMET, PPEItem.SAFETY_HELMET, PPEItem.SAFETY_SHOES]), # A -> AB 선호 반영
        HazardPPE(hazard=HazardFactor.NOISE, ppe=[PPEItem.HEARING_PROTECTION]),
        HazardPPE(hazard=HazardFactor.VIBRATION, ppe=[PPEItem.ANTI_VIBRATION_GLOVES]),
        HazardPPE(hazard=HazardFactor.DUST, ppe=[PPEItem.DUST_MASK, PPEItem.EYE_PROTECTION]),
        # [Add] 회전체 끼임 구체화
        HazardPPE(hazard=HazardFactor.ENTANGLEMENT_CRUSHING_FROM_ROTATING_MACHINERY, ppe=[PPEItem.ANTI_PINCH_GLOVES, PPEItem.PROTECTIVE_CLOTHING, PPEItem.SAFETY_SHOES]),
    ],
    # -------------------------
    # W02: Drilling/Demolition
    # -------------------------
    WorkEnvironment.DRILLING_CRUSHING_DEMOLITION_DISMANTLING: [
        HazardPPE(hazard=HazardFactor.DUST, ppe=[PPEItem.DUST_MASK, PPEItem.SAFETY_GOGGLES]),
        HazardPPE(hazard=HazardFactor.NOISE, ppe=[PPEItem.HEARING_PROTECTION]),
        HazardPPE(hazard=HazardFactor.VIBRATION, ppe=[PPEItem.ANTI_VIBRATION_GLOVES]),
        HazardPPE(hazard=HazardFactor.IMPACT, ppe=[PPEItem.AB_TYPE_SAFETY_HELMET, PPEItem.SAFETY_SHOES]),
        # [Add] 낙하물 위험
        HazardPPE(hazard=HazardFactor.FALLING_OBJECTS, ppe=[PPEItem.AB_TYPE_SAFETY_HELMET, PPEItem.SAFETY_SHOES]),
    ],
    # -------------------------
    # W03: Welding/Hot Work
    # -------------------------
    WorkEnvironment.WELDING_CUTTING_HOT_WORK: [
        HazardPPE(hazard=HazardFactor.LIGHTING, ppe=[PPEItem.WELDING_MASK, PPEItem.FACE_SHIELD_OR_WELDING_GOGGLES]),
        HazardPPE(hazard=HazardFactor.HIGH_TEMPERATURE, ppe=[PPEItem.HEAT_RESISTANT_MASK, PPEItem.HEAT_RESISTANT_GLOVES, PPEItem.HEAT_RESISTANT_CLOTHING, PPEItem.HEAT_RESISTANT_SHOES]),
        HazardPPE(hazard=HazardFactor.IMPACT, ppe=[PPEItem.FACE_SHIELD, PPEItem.AB_TYPE_SAFETY_HELMET, PPEItem.HEAT_RESISTANT_SHOES]), # Modify 반영: Safety Shoes -> Heat Resistant Shoes
        HazardPPE(hazard=HazardFactor.CONFINED_SPACE, ppe=[PPEItem.AIR_SUPPLIED_RESPIRATOR]),
        # [Add] 핫스파크 및 케미컬 에어로졸(흄)
        HazardPPE(hazard=HazardFactor.HOT_SPARKS, ppe=[PPEItem.WELDING_MASK, PPEItem.HEAT_RESISTANT_CLOTHING, PPEItem.HEAT_RESISTANT_GLOVES]),
        HazardPPE(hazard=HazardFactor.CHEMICAL_AEROSOL, ppe=[PPEItem.GAS_RESPIRATOR_MASK, PPEItem.RESPIRATOR_CARTRIDGE_FILTER]),
    ],
    # -------------------------
    # W04: Chemical Handling
    # -------------------------
    WorkEnvironment.CHEMICAL_HANDLING_MIXING_PAINTING_CLEANING_COATING: [
        HazardPPE(hazard=HazardFactor.TOXIC_SUBSTANCE, ppe=[PPEItem.GAS_RESPIRATOR_MASK, PPEItem.IMPERMEABLE_PROTECTIVE_SUIT, PPEItem.CHEMICAL_RESISTANT_GLOVES, PPEItem.CHEMICAL_RESISTANT_SAFETY_BOOTS]),
        HazardPPE(hazard=HazardFactor.CONFINED_SPACE, ppe=[PPEItem.AIR_SUPPLIED_RESPIRATOR, PPEItem.IMPERMEABLE_PROTECTIVE_SUIT]),
        # [Add] 구체적 화학 노출
        HazardPPE(hazard=HazardFactor.CHEMICAL_EXPOSURE_HAZARD, ppe=[PPEItem.CHEMICAL_RESISTANT_GLOVES, PPEItem.IMPERMEABLE_PROTECTIVE_SUIT, PPEItem.SAFETY_GOGGLES]),
        HazardPPE(hazard=HazardFactor.CHEMICAL_AEROSOL, ppe=[PPEItem.GAS_RESPIRATOR_MASK, PPEItem.SAFETY_GOGGLES]),
    ],
    # -------------------------
    # W05: Powder/Cement (General Dust)
    # -------------------------
    WorkEnvironment.POWDER_CEMENT_WOODWORKING_ABRASIVE_DUST: [
        HazardPPE(hazard=HazardFactor.DUST, ppe=[PPEItem.DUST_MASK, PPEItem.SAFETY_GOGGLES]),
        HazardPPE(hazard=HazardFactor.IMPACT, ppe=[PPEItem.A_TYPE_SAFETY_HELMET, PPEItem.SAFETY_SHOES]),
    ],
    # -------------------------
    # W06: Electrical
    # -------------------------
    WorkEnvironment.ELECTRICAL_INSTALLATION_INSPECTION_MAINTENANCE: [
        HazardPPE(hazard=HazardFactor.ELECTRIC_SHOCK, ppe=[PPEItem.INSULATING_HELMET, PPEItem.INSULATED_GLOVES, PPEItem.INSULATING_BOOTS, PPEItem.DIELECTRIC_TOOLS]),
        HazardPPE(hazard=HazardFactor.CAUGHT_IN_OR_BETWEEN, ppe=[PPEItem.SAFETY_SHOES, PPEItem.PROTECTIVE_CLOTHING]),
        # [Add] 고소 작업 병행 가능성 (Modify 반영: +Working at Height)
        HazardPPE(hazard=HazardFactor.FALL, ppe=[PPEItem.SAFETY_HARNESS, PPEItem.INSULATING_HELMET]),
    ],
    # -------------------------
    # W07: Working at Height
    # -------------------------
    WorkEnvironment.WORKING_AT_HEIGHT_SCAFFOLD_LADDER_OPENINGS_TEMPORARY_WALKWAYS: [
        HazardPPE(hazard=HazardFactor.FALL, ppe=[PPEItem.SAFETY_HARNESS, PPEItem.AB_TYPE_SAFETY_HELMET]),
        HazardPPE(hazard=HazardFactor.IMPACT, ppe=[PPEItem.AB_TYPE_SAFETY_HELMET, PPEItem.SAFETY_SHOES]), # A -> AB 반영
        HazardPPE(hazard=HazardFactor.SLIP, ppe=[PPEItem.SAFETY_SHOES, PPEItem.SAFETY_FOOTWEAR]),
        # [Add] 낙하물
        HazardPPE(hazard=HazardFactor.FALLING_OBJECTS, ppe=[PPEItem.AB_TYPE_SAFETY_HELMET, PPEItem.SAFETY_SHOES]),
    ],
    # -------------------------
    # W08: Lifting/Crane
    # -------------------------
    WorkEnvironment.LIFTING_LOADING_UNLOADING_CRANE: [
        HazardPPE(hazard=HazardFactor.IMPACT, ppe=[PPEItem.AB_TYPE_SAFETY_HELMET, PPEItem.SAFETY_SHOES]),
        HazardPPE(hazard=HazardFactor.CAUGHT_IN_OR_BETWEEN, ppe=[PPEItem.CUT_AND_ANTI_PINCH_GLOVES, PPEItem.SAFETY_SHOES]),
        HazardPPE(hazard=HazardFactor.FALLING_OBJECTS, ppe=[PPEItem.AB_TYPE_SAFETY_HELMET, PPEItem.SAFETY_SHOES]),
    ],
    # -------------------------
    # W09: Mobile Equipment
    # -------------------------
    WorkEnvironment.WORKING_NEAR_MOBILE_EQUIPMENT: [
        HazardPPE(hazard=HazardFactor.CAUGHT_IN_OR_BETWEEN, ppe=[PPEItem.HIGH_VISIBILITY_VEST, PPEItem.SAFETY_SHOES]), # High Vis 추가
        HazardPPE(hazard=HazardFactor.IMPACT, ppe=[PPEItem.HARD_HAT, PPEItem.SAFETY_SHOES]),
        HazardPPE(hazard=HazardFactor.SLIP, ppe=[PPEItem.SAFETY_SHOES]),
    ],
    # -------------------------
    # W10: Rotating Machinery
    # -------------------------
    WorkEnvironment.ROTATING_MACHINERY_PRESS_CONVEYOR_EQUIPMENT: [
        HazardPPE(hazard=HazardFactor.CAUGHT_IN_OR_BETWEEN, ppe=[PPEItem.CUT_AND_ANTI_PINCH_GLOVES, PPEItem.SAFETY_SHOES, PPEItem.PROTECTIVE_CLOTHING]),
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
    ],
    # -------------------------
    # W12: Cold Work
    # -------------------------
    WorkEnvironment.LOW_TEMPERATURE_COLD_WORK: [
        HazardPPE(hazard=HazardFactor.LOW_TEMPERATURE, ppe=[PPEItem.COLD_PROTECTION_HOOD, PPEItem.COLD_PROTECTION_CLOTHING, PPEItem.INSULATING_COLD_BOOTS, PPEItem.COLD_PROTECTION_GLOVES]),
        HazardPPE(hazard=HazardFactor.SLIP, ppe=[PPEItem.SAFETY_SHOES]),
        # [Add] 결빙 조건
        HazardPPE(hazard=HazardFactor.ICE_CONDITIONS, ppe=[PPEItem.SAFETY_SHOES, PPEItem.SAFETY_FOOTWEAR]), # 아이젠 등은 Safety footwear에 포함 가정
    ],
    # -------------------------
    # [New] W13: Healthcare / Biological (분석결과 대량 발생)
    # -------------------------
    WorkEnvironment.HEALTHCARE_PATIENT_CARE_INFECTIOUS: [
        HazardPPE(hazard=HazardFactor.BIOLOGICAL_INFECTIOUS, ppe=[PPEItem.MEDICAL_GLOVES, PPEItem.PROTECTIVE_CLOTHING, PPEItem.FACE_SHIELD, PPEItem.RESPIRATOR_CARTRIDGE_FILTER]),
        HazardPPE(hazard=HazardFactor.CHEMICAL_EXPOSURE_HAZARD, ppe=[PPEItem.MEDICAL_GLOVES, PPEItem.SAFETY_GOGGLES]),
    ],
    # -------------------------
    # [New] W14: Tunnel / Underground (분석결과 290건)
    # -------------------------
    WorkEnvironment.TUNNEL_UNDERGROUND_WORK: [
        HazardPPE(hazard=HazardFactor.FALLING_OBJECTS, ppe=[PPEItem.HARD_HAT, PPEItem.SAFETY_HELMET, PPEItem.SAFETY_SHOES]),
        HazardPPE(hazard=HazardFactor.DUST, ppe=[PPEItem.DUST_MASK, PPEItem.EYE_PROTECTION]),
        HazardPPE(hazard=HazardFactor.SLIP, ppe=[PPEItem.SAFETY_SHOES]),
        HazardPPE(hazard=HazardFactor.LIGHTING, ppe=[PPEItem.HIGH_VISIBILITY_VEST, PPEItem.SAFETY_HELMET]), # 시인성 확보
        HazardPPE(hazard=HazardFactor.IMPACT, ppe=[PPEItem.HARD_HAT, PPEItem.SAFETY_SHOES]),
    ],
    # -------------------------
    # [New] W15: Woodworking Specific (분석결과 336건)
    # -------------------------
    WorkEnvironment.WOODWORKING_SPECIFIC: [
        HazardPPE(hazard=HazardFactor.DUST, ppe=[PPEItem.DUST_MASK, PPEItem.SAFETY_GOGGLES]),
        HazardPPE(hazard=HazardFactor.CUTTING, ppe=[PPEItem.CUT_RESISTANT_GLOVES, PPEItem.SAFETY_GOGGLES]),
        HazardPPE(hazard=HazardFactor.NOISE, ppe=[PPEItem.HEARING_PROTECTION]),
        HazardPPE(hazard=HazardFactor.IMPACT, ppe=[PPEItem.SAFETY_SHOES]),
    ],
    # -------------------------
    # [New] W16: Rooftop Solar (분석결과 34건)
    # -------------------------
    WorkEnvironment.ROOFTOP_SOLAR_INSTALLATION: [
        HazardPPE(hazard=HazardFactor.FALL, ppe=[PPEItem.SAFETY_HARNESS, PPEItem.AB_TYPE_SAFETY_HELMET]),
        HazardPPE(hazard=HazardFactor.ELECTRIC_SHOCK, ppe=[PPEItem.INSULATED_GLOVES, PPEItem.INSULATING_BOOTS, PPEItem.DIELECTRIC_TOOLS]),
        HazardPPE(hazard=HazardFactor.SLIP, ppe=[PPEItem.SAFETY_SHOES]),
        HazardPPE(hazard=HazardFactor.HIGH_TEMPERATURE, ppe=[PPEItem.PROTECTIVE_CLOTHING, PPEItem.SAFETY_GOGGLES]), # 야외 태양광
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
