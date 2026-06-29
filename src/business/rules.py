
from typing import Dict

# Acciones posibles de negocio
ACTION_APPROVE = "approve"
ACTION_FLAG = "flag_for_review"
ACTION_MANUAL = "manual_review"


def decide(is_healthy: bool, confidence: float, threshold: float) -> Dict:
    """Decide la acción según el veredicto del modelo y su confianza.

    - confianza por debajo del umbral -> revisión manual (zona gris, no confiamos),
    - sano con confianza suficiente    -> aprobar,
    - enfermo con confianza suficiente -> marcar para revisión/acción.

    La 'zona gris' es clave en agro: ante baja confianza preferimos escalar a un
    humano antes que dejar pasar una planta enferma (falso negativo costoso).
    """
    if confidence < threshold:
        action = ACTION_MANUAL
        reason = f"Confianza {confidence:.2f} por debajo del umbral {threshold:.2f}"
    elif is_healthy:
        action = ACTION_APPROVE
        reason = "Planta sana con confianza suficiente"
    else:
        action = ACTION_FLAG
        reason = "Condición no saludable detectada con confianza suficiente"

    return {
        "action": action,
        "reason": reason,
        "threshold_applied": round(threshold, 2),
    }


def apply_business_decision(result: Dict, threshold: float) -> Dict:
    """Extrae veredicto + confianza del resultado de inferencia y añade la decisión.

    Funciona tanto para imagen (usa prediction) como para video (usa summary).
    """
    if result.get("input_type") == "video":
        is_healthy = result["summary"]["is_healthy"]
        confidence = result["summary"]["avg_confidence"]
    else:
        is_healthy = result["prediction"]["is_healthy"]
        confidence = result["prediction"]["confidence"]

    result["business_decision"] = decide(is_healthy, confidence, threshold)
    return result
