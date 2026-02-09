# params/structure.py

from enum import Enum


class WorkEnvironment(Enum):
    # --- 기존 작업환경 ---
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

    # --- [New] 제안된 작업환경 추가 ---
    HEALTHCARE_PATIENT_CARE = "Healthcare, patient care, infectious disease handling"
    TUNNEL_UNDERGROUND_WORK = "Tunnel, underground excavation work"
    ROOFTOP_SOLAR_INSTALLATION = "Rooftop solar panel installation"  # 고소작업+전기의 복합 성격


class HazardFactor(Enum):
    # --- 기존 위험요소 ---
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

    # --- [New] 제안된 위험요소 추가 (구체화된 항목들) ---
    CHEMICAL_EXPOSURE_HAZARD = "Chemical exposure hazard" # TOXIC_SUBSTANCE와 유사하나 제안된 용어 반영
    CHEMICAL_AEROSOL = "Chemical aerosol"
    FALLING_OBJECTS = "Falling objects"  # FALL(사람의 추락)과 구분되는 낙하물
    ICE_CONDITIONS = "Ice conditions"    # SLIP의 구체적 원인
    HOT_SPARKS = "Hot sparks"            # 용접 시 구체적 위험
    ENTANGLEMENT_CRUSHING_FROM_ROTATING_MACHINERY = "Entanglement/crushing from rotating machinery" # CAUGHT_IN의 구체화


class PPEItem(Enum):
    # --- 기존 PPE ---
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

    # --- [New] 제안된 PPE 추가 ---
    # 제안된 일반화된 용어 및 신규 아이템
    SAFETY_HELMET = "Safety helmet"  # A/AB/ABE 타입을 포괄하는 일반 용어
    FACE_SHIELD_OR_WELDING_GOGGLES = "Face shield or welding goggles"
    MEDICAL_GLOVES = "Medical gloves"
    PROTECTIVE_CLOTHING = "Protective clothing" # 일반 작업복/보호복
    SAFETY_FOOTWEAR = "Safety footwear"         # 안전화 포괄 용어
    INSULATING_COLD_BOOTS = "Insulating cold boots"
    DIELECTRIC_TOOLS = "Dielectric tools"       # 절연 공구
