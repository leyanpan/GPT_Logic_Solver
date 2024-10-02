from sops import SOp
from typing import List

def sop_generate(prompt_tokens: List[str], id_to_token: List[str], sop: SOp, stop_tokens: List[str] = ["SAT", "UNSAT"], max_generation_length: int = 800) -> List[str]:
    """
    Generate a sequence of tokens from a prompt using a given SOp model.

    Parameters:
        prompt_tokens: The input prompt tokens.
        id_to_token: The mapping from token ids to token strings.
        sop: The SOp model to use for generation.

    Returns:
        generated_tokens: The generated sequence of tokens.
    """
    assert sop.dim == len(id_to_token), "The SOp model must have the same dimension as the token vocabulary."
    generated_tokens = prompt_tokens.copy()
    for _ in range(max_generation_length):
        sop_output = sop.abstract_eval(generated_tokens)
        next_token_id = sop_output[-1, :].argmax().item()
        next_token = id_to_token[next_token_id]
        generated_tokens.append(next_token)

        if next_token in stop_tokens:
            break

    return generated_tokens

