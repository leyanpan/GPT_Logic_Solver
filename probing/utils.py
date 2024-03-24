def task_ids(pred_str, old_tokenizer=False):
    if old_tokenizer:
        pred_str = pred_str.replace('-', '- ')
    tokens = pred_str.split()
    task_ids_list = []
    if '[SEP]' in pred_str:
        sep_idx = tokens.index('[SEP]')
        task_ids_list = [-1] * (sep_idx + 1)
        tokens = tokens[sep_idx + 1:]
    assuming = False
    for token in tokens:
        if token == '0':
            task_ids_list.append(0)
            assuming = False
        if token == '(':
            task_ids_list.append(1)
            assuming = True
        if token == ')':
            task_ids_list.append(2)
            assuming = False
        if token == 'SAT' or token == 'UNSAT':
            task_ids_list.append(5)
        else:
            if assuming:
                task_ids_list.append(3)
            else:
                task_ids_list.append(4)

    return task_ids_list
        