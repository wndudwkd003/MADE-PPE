# params/ppe_mapping.py
# 작업환경 -> 위험요소 -> 필수 PPE(리스트) 매핑

from dataclasses import dataclass


from params.structure import WorkEnvironment, HazardFactor, PPEItem


@dataclass(frozen=True)
class HazardPPE:
    hazard: HazardFactor
    ppe: list[PPEItem]


# WorkEnvironment별 HazardPPE 리스트(= hazard -> ppe) 구조
WORK_ENV_TO_HAZARD_PPE: dict[WorkEnvironment, list[HazardPPE]] = {
    # -------------------------
    # W01
    # -------------------------
    WorkEnvironment.W01_CUTTING_GRINDING_MACHINING_POLISHING: [
        HazardPPE(
            hazard=HazardFactor.CUT,
            ppe=[
                PPEItem.SAFETY_GOGGLES,
                PPEItem.SAFETY_SHOES,
                PPEItem.CUT_RESISTANT_GLOVES,
                PPEItem.FACE_SHIELD,
            ],
        ),
        HazardPPE(
            hazard=HazardFactor.IMPACT,
            ppe=[
                PPEItem.A_TYPE_SAFETY_HELMET,
                PPEItem.SAFETY_SHOES,
            ],
        ),
        HazardPPE(
            hazard=HazardFactor.NOISE,
            ppe=[PPEItem.HEARING_PROTECTION],
        ),
        HazardPPE(
            hazard=HazardFactor.VIBRATION,
            ppe=[PPEItem.ANTI_VIBRATION_GLOVES],
        ),
        HazardPPE(
            hazard=HazardFactor.DUST,
            ppe=[PPEItem.DUST_MASK],
        ),
    ],
    # -------------------------
    # W02
    # -------------------------
    WorkEnvironment.W02_DRILLING_CRUSHING_DEMOLITION_DISMANTLING: [
        HazardPPE(hazard=HazardFactor.DUST, ppe=[PPEItem.DUST_MASK]),
        HazardPPE(hazard=HazardFactor.NOISE, ppe=[PPEItem.HEARING_PROTECTION]),
        HazardPPE(hazard=HazardFactor.VIBRATION, ppe=[PPEItem.ANTI_VIBRATION_GLOVES]),
        HazardPPE(
            hazard=HazardFactor.IMPACT,
            ppe=[PPEItem.A_TYPE_SAFETY_HELMET, PPEItem.SAFETY_SHOES],
        ),
    ],
    # -------------------------
    # W03
    # -------------------------
    WorkEnvironment.W03_WELDING_CUTTING_HOT_WORK: [
        HazardPPE(
            hazard=HazardFactor.LIGHTING,  # 아크광/강렬한 광
            ppe=[PPEItem.WELDING_MASK],
        ),
        HazardPPE(
            hazard=HazardFactor.HIGH_TEMPERATURE,
            ppe=[
                PPEItem.HEAT_RESISTANT_MASK,
                PPEItem.HEAT_RESISTANT_GLOVES,
                PPEItem.HEAT_RESISTANT_CLOTHING,
                PPEItem.HEAT_RESISTANT_SHOES,
            ],
        ),
        HazardPPE(
            hazard=HazardFactor.IMPACT,  # 불꽃/비산물/스패터를 충돌/비산으로 묶음
            ppe=[
                PPEItem.FACE_SHIELD,
                PPEItem.A_TYPE_SAFETY_HELMET,
                PPEItem.SAFETY_SHOES,
            ],
        ),
        HazardPPE(
            hazard=HazardFactor.CONFINED_SPACE,  # 밀폐에서의 용접/용단
            ppe=[
                PPEItem.AIR_SUPPLIED_RESPIRATOR,  # 송기마스크/공기호흡기
            ],
        ),
    ],
    # -------------------------
    # W04
    # -------------------------
    WorkEnvironment.W04_CHEMICAL_HANDLING_MIXING_PAINTING_CLEANING_COATING: [
        HazardPPE(
            hazard=HazardFactor.TOXIC_SUBSTANCE,
            ppe=[
                PPEItem.GAS_RESPIRATOR_MASK,
                PPEItem.IMPERMEABLE_PROTECTIVE_SUIT,
                PPEItem.CHEMICAL_RESISTANT_GLOVES,
                PPEItem.CHEMICAL_RESISTANT_SAFETY_BOOTS,
                PPEItem.RESPIRATOR_CARTRIDGE_FILTER,
            ],
        ),
        HazardPPE(
            hazard=HazardFactor.CONFINED_SPACE,
            ppe=[PPEItem.AIR_SUPPLIED_RESPIRATOR],
        ),
    ],
    # -------------------------
    # W05
    # -------------------------
    WorkEnvironment.W05_POWDER_CEMENT_WOODWORKING_ABRASIVE_DUST: [
        HazardPPE(hazard=HazardFactor.DUST, ppe=[PPEItem.DUST_MASK]),
        HazardPPE(
            hazard=HazardFactor.IMPACT,
            ppe=[PPEItem.A_TYPE_SAFETY_HELMET, PPEItem.SAFETY_SHOES],
        ),
    ],
    # -------------------------
    # W06
    # -------------------------
    WorkEnvironment.W06_ELECTRICAL_INSTALLATION_INSPECTION_MAINTENANCE: [
        HazardPPE(
            hazard=HazardFactor.ELECTRIC_SHOCK,
            ppe=[
                PPEItem.INSULATING_HELMET,
                PPEItem.INSULATED_GLOVES,
                PPEItem.INSULATING_BOOTS,
                PPEItem.ABE_TYPE_SAFETY_HELMET,
            ],
        ),
        HazardPPE(
            hazard=HazardFactor.CAUGHT_IN_OR_BETWEEN,  # 끼임/충격을 한 항목으로
            ppe=[PPEItem.SAFETY_SHOES],
        ),
    ],
    # -------------------------
    # W07
    # -------------------------
    WorkEnvironment.W07_WORKING_AT_HEIGHT_SCAFFOLD_LADDER_OPENINGS_TEMPORARY_WALKWAYS: [
        HazardPPE(
            hazard=HazardFactor.FALL,
            ppe=[PPEItem.SAFETY_HARNESS, PPEItem.AB_TYPE_SAFETY_HELMET],
        ),
        HazardPPE(
            hazard=HazardFactor.IMPACT,
            ppe=[PPEItem.A_TYPE_SAFETY_HELMET, PPEItem.SAFETY_SHOES],
        ),
        HazardPPE(
            hazard=HazardFactor.SLIP,
            ppe=[PPEItem.SAFETY_SHOES],  # "전화(미끄럼방지)"를 안전화로 정규화
        ),
    ],
    # -------------------------
    # W08
    # -------------------------
    WorkEnvironment.W08_LIFTING_LOADING_UNLOADING_CRANE: [
        HazardPPE(
            hazard=HazardFactor.IMPACT,
            ppe=[PPEItem.A_TYPE_SAFETY_HELMET, PPEItem.SAFETY_SHOES],
        ),
        HazardPPE(
            hazard=HazardFactor.CAUGHT_IN_OR_BETWEEN,
            ppe=[PPEItem.ANTI_PINCH_GLOVES, PPEItem.SAFETY_SHOES],
        ),
    ],
    # -------------------------
    # W09
    # -------------------------
    WorkEnvironment.W09_WORKING_NEAR_MOBILE_EQUIPMENT: [
        HazardPPE(
            hazard=HazardFactor.CAUGHT_IN_OR_BETWEEN,
            ppe=[PPEItem.ANTI_PINCH_GLOVES, PPEItem.SAFETY_SHOES],
        ),
        HazardPPE(
            hazard=HazardFactor.IMPACT,
            ppe=[PPEItem.A_TYPE_SAFETY_HELMET, PPEItem.SAFETY_SHOES],
        ),
        HazardPPE(
            hazard=HazardFactor.SLIP,
            ppe=[PPEItem.SAFETY_SHOES],  # 미끄럼방지 작업화/안전화 -> 안전화로 정규화
        ),
    ],
    # -------------------------
    # W10
    # -------------------------
    WorkEnvironment.W10_ROTATING_MACHINERY_PRESS_CONVEYOR_EQUIPMENT: [
        HazardPPE(
            hazard=HazardFactor.CAUGHT_IN_OR_BETWEEN,
            ppe=[PPEItem.ANTI_PINCH_GLOVES, PPEItem.SAFETY_SHOES],
        ),
        HazardPPE(
            hazard=HazardFactor.CUT,
            ppe=[PPEItem.CUT_RESISTANT_GLOVES, PPEItem.SAFETY_GOGGLES],
        ),
    ],
    # -------------------------
    # W11
    # -------------------------
    WorkEnvironment.W11_CONFINED_SPACE_WORK: [
        HazardPPE(
            hazard=HazardFactor.CONFINED_SPACE,
            ppe=[PPEItem.AIR_SUPPLIED_RESPIRATOR],
        ),
        HazardPPE(
            hazard=HazardFactor.FALL,
            ppe=[PPEItem.SAFETY_HARNESS, PPEItem.AIR_SUPPLIED_RESPIRATOR],
        ),
        HazardPPE(
            hazard=HazardFactor.TOXIC_SUBSTANCE,
            ppe=[PPEItem.AIR_SUPPLIED_RESPIRATOR, PPEItem.GAS_RESPIRATOR_MASK],
        ),
    ],
    # -------------------------
    # W12
    # -------------------------
    WorkEnvironment.W12_LOW_TEMPERATURE_COLD_WORK: [
        HazardPPE(
            hazard=HazardFactor.LOW_TEMPERATURE,
            ppe=[
                PPEItem.COLD_PROTECTION_HOOD,
                PPEItem.COLD_PROTECTION_CLOTHING,
                PPEItem.COLD_PROTECTION_BOOTS,
                PPEItem.COLD_PROTECTION_GLOVES,
            ],
        ),
        HazardPPE(
            hazard=HazardFactor.SLIP,
            ppe=[PPEItem.SAFETY_SHOES],  # 미끄럼방지 작업화/안전화 -> 안전화로 정규화
        ),
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
