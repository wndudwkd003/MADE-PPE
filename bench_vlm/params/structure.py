# benchmark_vlm/params/structure.py

from enum import Enum

class WorkEnvironment(Enum):
    # --- [Existing] 기존 작업환경 ---
    CUTTING_GRINDING_MACHINING_POLISHING = "Cutting, grinding, machining, polishing"
    DRILLING_CRUSHING_DEMOLITION_DISMANTLING = "Drilling, crushing, demolition, dismantling"
    WELDING_CUTTING_HOT_WORK = "Welding, cutting, hot work"
    CHEMICAL_HANDLING_MIXING_PAINTING_CLEANING_COATING = "Chemical handling, mixing, painting, cleaning, coating"
    POWDER_CEMENT_WOODWORKING_ABRASIVE_DUST = "Powder, cement, woodworking, abrasive dust"
    ELECTRICAL_INSTALLATION_INSPECTION_MAINTENANCE = "Electrical installation, inspection, maintenance"
    WORKING_AT_HEIGHT_SCAFFOLD_LADDER_OPENINGS_TEMPORARY_WALKWAYS = "Working at height, scaffolding, ladders, openings, temporary walkways"
    LIFTING_LOADING_UNLOADING_CRANE = "Lifting, loading/unloading, crane operations"
    WORKING_NEAR_MOBILE_EQUIPMENT = "Working near mobile equipment"
    ROTATING_MACHINERY_PRESS_CONVEYOR_EQUIPMENT = "Rotating machinery, presses, conveyors, equipment work"
    CONFINED_SPACE_WORK = "Confined space work"
    LOW_TEMPERATURE_COLD_WORK = "Low-temperature, cold work"

    # --- [New] 분석 결과 기반 추가 (빈도수 높은 항목 반영) ---
    # 의료/바이오 관련 (통합)
    HEALTHCARE_PATIENT_CARE_INFECTIOUS = "Healthcare, patient care, infectious disease, lab"
    # 터널/지하 작업
    TUNNEL_UNDERGROUND_WORK = "Tunnel, underground excavation work"
    # 목공 (기존 W05와 분리되어 식별됨)
    WOODWORKING_SPECIFIC = "Woodworking, timber processing"
    # 태양광/지붕 작업
    ROOFTOP_SOLAR_INSTALLATION = "Rooftop solar panel installation"


class HazardFactor(Enum):
    # --- [Existing] 기존 위험요소 ---
    CUTTING = "Cutting"
    TOXIC_SUBSTANCE = "Toxic substances"
    HIGH_TEMPERATURE = "High temperature"
    LOW_TEMPERATURE = "Low temperature"
    FALL = "Fall"
    DUST = "Dust"
    IMPACT = "Impact"
    NOISE = "Noise"
    VIBRATION = "Vibration"
    ELECTRIC_SHOCK = "Electric shock"
    CONFINED_SPACE = "Confined space"
    SLIP = "Slip"
    CAUGHT_IN_OR_BETWEEN = "Caught-in/between"
    LIGHTING = "Lighting"
    BIOLOGICAL_INFECTIOUS = "Biological/infectious"

    # --- [New] 분석 결과 기반 추가 (구체화된 위험) ---
    CHEMICAL_EXPOSURE_HAZARD = "Chemical exposure hazard" # 피부 접촉 등 구체적 위험
    CHEMICAL_AEROSOL = "Chemical aerosol"                 # 흡입 위험
    FALLING_OBJECTS = "Falling objects"                   # 낙하물 (Fall은 사람 추락, 이건 물체 낙하)
    ICE_CONDITIONS = "Ice conditions"                     # 결빙으로 인한 미끄러짐
    HOT_SPARKS = "Hot sparks"                             # 용접 스파크
    ENTANGLEMENT_CRUSHING_FROM_ROTATING_MACHINERY = "Entanglement/crushing from rotating machinery" # 끼임의 구체화


class PPEItem(Enum):
    # --- [Existing] 기존 PPE ---
    SAFETY_GOGGLES = "Safety goggles"
    SAFETY_SHOES = "Safety shoes"
    CUT_RESISTANT_GLOVES = "Cut-resistant gloves"
    FACE_SHIELD = "Face shield"
    GAS_RESPIRATOR_MASK = "Gas respirator mask"
    IMPERMEABLE_PROTECTIVE_SUIT = "Impermeable protective suit"
    CHEMICAL_RESISTANT_GLOVES = "Chemical-resistant gloves"
    CHEMICAL_RESISTANT_SAFETY_BOOTS = "Chemical-resistant safety boots"

    RESPIRATOR_CARTRIDGE_FILTER = "Respirator cartridge/filter"
    HEAT_RESISTANT_MASK = "Heat-resistant mask"
    HEAT_RESISTANT_GLOVES = "Heat-resistant gloves"
    HEAT_RESISTANT_CLOTHING = "Heat-resistant clothing"
    HEAT_RESISTANT_SHOES = "Heat-resistant shoes"
    COLD_PROTECTION_HOOD = "Cold-protection hood"
    COLD_PROTECTION_BOOTS = "Cold-protection boots"
    COLD_PROTECTION_GLOVES = "Cold-protection gloves"

    COLD_PROTECTION_CLOTHING = "Cold-protection clothing"
    SAFETY_HARNESS = "Safety harness"
    AB_TYPE_SAFETY_HELMET = "AB-type safety helmet"
    DUST_MASK = "Dust mask"
    A_TYPE_SAFETY_HELMET = "A-type safety helmet"
    HEARING_PROTECTION = "Hearing protection"
    ANTI_VIBRATION_GLOVES = "Anti-vibration gloves"
    INSULATED_GLOVES = "Insulated gloves"

    INSULATING_HELMET = "Insulating helmet"
    INSULATING_BOOTS = "Insulating boots"
    ABE_TYPE_SAFETY_HELMET = "ABE-type safety helmet"
    AIR_SUPPLIED_RESPIRATOR = "Air-supplied respirator"
    ANTI_PINCH_GLOVES = "Anti-pinch gloves"
    WELDING_MASK = "Welding mask"

    # --- [New] 분석 결과 기반 추가 ---
    SAFETY_HELMET = "Safety helmet"  # 일반형 헬멧 (분석에서 빈출)
    FACE_SHIELD_OR_WELDING_GOGGLES = "Face shield or welding goggles"
    MEDICAL_GLOVES = "Medical gloves"
    PROTECTIVE_CLOTHING = "Protective clothing" # 일반 작업복/보호복
    SAFETY_FOOTWEAR = "Safety footwear"         # 안전화 통칭
    INSULATING_COLD_BOOTS = "Insulating cold boots"
    DIELECTRIC_TOOLS = "Dielectric tools"       # 절연 공구 (안전장비의 일종으로 분류)
    HIGH_VISIBILITY_VEST = "High visibility vest" # 터널/건설 현장 필수
    EYE_PROTECTION = "Eye protection"           # 보안경 통칭
    HARD_HAT = "Hard hat"                       # Safety Helmet의 이명
    CUT_AND_ANTI_PINCH_GLOVES = "Cut and anti-pinch gloves" # 복합 기능 장갑
