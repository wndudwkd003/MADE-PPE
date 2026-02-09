# params/structure.py

from enum import Enum


class WorkEnvironment(Enum):
    W01_CUTTING_GRINDING_MACHINING_POLISHING = "Cutting, grinding, machining, polishing"
    W02_DRILLING_CRUSHING_DEMOLITION_DISMANTLING = (
        "Drilling, crushing, demolition, dismantling"
    )
    W03_WELDING_CUTTING_HOT_WORK = "Welding, cutting, hot work"
    W04_CHEMICAL_HANDLING_MIXING_PAINTING_CLEANING_COATING = (
        "Chemical handling, mixing, painting, cleaning, coating"
    )
    W05_POWDER_CEMENT_WOODWORKING_ABRASIVE_DUST = (
        "Powder, cement, woodworking, abrasive dust"
    )
    W06_ELECTRICAL_INSTALLATION_INSPECTION_MAINTENANCE = (
        "Electrical installation, inspection, maintenance"
    )
    W07_WORKING_AT_HEIGHT_SCAFFOLD_LADDER_OPENINGS_TEMPORARY_WALKWAYS = (
        "Working at height, scaffolding, ladders, openings, temporary walkways"
    )
    W08_LIFTING_LOADING_UNLOADING_CRANE = "Lifting, loading/unloading, crane operations"
    W09_WORKING_NEAR_MOBILE_EQUIPMENT = "Working near mobile equipment"
    W10_ROTATING_MACHINERY_PRESS_CONVEYOR_EQUIPMENT = (
        "Rotating machinery, presses, conveyors, equipment work"
    )
    W11_CONFINED_SPACE_WORK = "Confined space work"
    W12_LOW_TEMPERATURE_COLD_WORK = "Low-temperature, cold work"


class HazardFactor(Enum):
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


class PPEItem(Enum):
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
