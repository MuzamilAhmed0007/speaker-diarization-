# utils/eval.py

def compute_metrics(pred, label):
    """
    Compute DER, FA, and MD.

    Args:
        pred (Tensor): Predicted speaker labels [T]
        label (Tensor): Ground truth speaker labels [T]

    Returns:
        tuple: (DER, FA, MD)
    """
    T_total = len(label)

    # Ensure same length
    if len(pred) != T_total:
        min_len = min(len(pred), len(label))
        pred = pred[:min_len]
        label = label[:min_len]
        T_total = min_len

    T_FA = 0  # False alarm duration
    T_MD = 0  # Missed detection duration
    T_SC = 0  # Speaker confusion

    for p, l in zip(pred, label):
        if l == -100:  # Silence
            if p != -100:
                T_FA += 1
        else:  # Speech
            if p == -100:
                T_MD += 1
            elif p != l:
                T_SC += 1

    DER = (T_FA + T_MD + T_SC) / T_total
    FA = T_FA / T_total
    MD = T_MD / T_total

    return DER, FA, MD


def print_metrics(DER, FA, MD):
    print(f"DER: {DER:.4f} | FA: {FA:.4f} | MD: {MD:.4f}")
