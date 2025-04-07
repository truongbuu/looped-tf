import torch
import torch.nn as nn

# for parity, sum reverse, copy, addition
def test_model(model, test_len, test_bs, generate_prompt_matrix, convert_to_one_hot, one_hot_to_int, exact_match_accuracy):
    with torch.no_grad(): 
        xs, batch_num, ys, mask = generate_prompt_matrix(test_bs, min_num_digits = test_len, max_num_digits = test_len+1, max_len = test_len+2)
        # 22 23 24 40
        xs = torch.tensor(convert_to_one_hot(xs))
        xs = xs.cuda()
        # ys = ys.cuda()
        states = model.looped_forward(xs, horizon = test_len+2)
        last = states[batch_num[0].item()-1]
        # last = states[-1]
        results = torch.tensor(one_hot_to_int(last.cpu().detach().numpy()))
        acc = exact_match_accuracy(results[mask==1].reshape(xs.shape[0], mask[0].sum()), ys[mask==1].reshape(xs.shape[0], mask[0].sum()))
    return acc

def test_model_adaptive(model, test_len, test_bs, generate_prompt_matrix, convert_to_one_hot, one_hot_to_int, exact_match_accuracy, extra = 20):
    loss_func = nn.CrossEntropyLoss()
    with torch.no_grad():
        xs, batch_num, ys, mask = generate_prompt_matrix(test_bs, min_num_digits = test_len, max_num_digits = test_len+1, max_len = test_len+2)
        # 22 23 24 40
        xs = torch.tensor(convert_to_one_hot(xs))
        xs = xs.cuda()
        # ys = ys.cuda()
        states = model.looped_forward(xs, test_len + extra)
        acc_list = []
        cross_entropy_list = []
        for step in range(batch_num[0].item()+extra):
            last = states[step]
            results = torch.tensor(one_hot_to_int(last.cpu().detach().numpy()))
            acc = exact_match_accuracy(results[mask==1].reshape(xs.shape[0], mask[0].sum()), ys[mask==1].reshape(xs.shape[0], mask[0].sum()))
            acc_loss = loss_func(last[mask == 1].cuda(), results[mask == 1].cuda())
            acc_list.append(acc)
            cross_entropy_list.append(acc_loss.item())
            # print("acc_list =", acc_list)
            # print("cross_entropy_list =", cross_entropy_list)
        index = torch.argmin(torch.tensor(cross_entropy_list))
        return acc_list[index], index

# multi

def test_model_multi(model, test_len, test_bs, generate_prompt_matrix, convert_to_one_hot, one_hot_to_int, exact_match_accuracy):
    with torch.no_grad():
        xs, batch_num, batch_num_1, ys, mask = generate_prompt_matrix(test_bs, min_num_digits = test_len, max_num_digits = test_len+1, max_len = test_len+2, test=True)
        # 22 23 24 40
        xs = torch.tensor(convert_to_one_hot(xs))
        xs = xs.cuda()
        # ys = ys.cuda()
        states = model.looped_forward(xs, horizon = test_len*2)
        last = states[batch_num[0].item()*batch_num_1[0].item()-1]
        results = torch.tensor(one_hot_to_int(last.cpu().detach().numpy()))
        acc = exact_match_accuracy(results[mask==1].reshape(xs.shape[0], mask[0].sum()), ys[mask==1].reshape(xs.shape[0], mask[0].sum()))
    return acc

def test_model_multi_adaptive(model, test_len, test_bs, generate_prompt_matrix, convert_to_one_hot, one_hot_to_int, exact_match_accuracy, extra = 30):
    with torch.no_grad():
        loss_func = nn.CrossEntropyLoss()
        xs, batch_num, batch_num_1, ys, mask = generate_prompt_matrix(test_bs, min_num_digits = test_len, max_num_digits = test_len+1, max_len = test_len+2, test=True)
        xs = torch.tensor(convert_to_one_hot(xs))
        xs = xs.cuda()
        acc_list = []
        cross_entropy_list = []
        states = model.looped_forward(xs, horizon = test_len*2+extra)
        for step in range(batch_num[0].item()*batch_num_1[0].item()+extra):
            last = states[step]
            results = torch.tensor(one_hot_to_int(last.cpu().detach().numpy()))
            acc = exact_match_accuracy(results[mask==1].reshape(xs.shape[0], mask[0].sum()), ys[mask==1].reshape(xs.shape[0], mask[0].sum()))
            loss = loss_func(last[mask==1].cuda(), results[mask==1].cuda())
            acc_list.append(acc)
            cross_entropy_list.append(loss.item())
        # print("acc_list =", acc_list)
        # print("cross_entropy_list =", cross_entropy_list)
        index = torch.argmin(torch.tensor(cross_entropy_list))
    return acc_list[index], index

# dict

def test_model_dict(model, test_len, test_bs, generate_prompt_matrix, convert_to_one_hot, one_hot_to_int, exact_match_accuracy):
    with torch.no_grad():
        xs, batch_num, ys, mask = generate_prompt_matrix(test_bs, min_num_digits = test_len, max_num_digits = test_len+1, max_len = test_len+2)
        # 22 23 24 40
        # xs = torch.tensor(convert_to_one_hot(xs))
        xs = xs.cuda()
        # ys = ys.cuda()
        states = model.looped_forward(xs, horizon = test_len+2)
        last = states[batch_num[0].item()-1]
        # last = states[-1]
        results = torch.tensor(one_hot_to_int(last.cpu().detach().numpy()))
        acc = exact_match_accuracy(results[mask==1].reshape(xs.shape[0], mask[0].sum()), ys[mask==1].reshape(xs.shape[0], mask[0].sum()))
    return acc

def test_model_dict_adaptive(model, test_len, test_bs, generate_prompt_matrix, convert_to_one_hot, one_hot_to_int, exact_match_accuracy, extra = 30):
    with torch.no_grad():
        loss_func = nn.CrossEntropyLoss()
        xs, batch_num, ys, mask = generate_prompt_matrix(test_bs, min_num_digits = test_len, max_num_digits = test_len+1, max_len = test_len+2)
        xs = xs.cuda()
        states = model.looped_forward(xs, horizon = test_len+2+extra)
        acc_list = []
        cross_entropy_list = []
        for step in range(batch_num[0].item()+extra):
            last = states[step]
            results = torch.tensor(one_hot_to_int(last.cpu().detach().numpy()))
            acc = exact_match_accuracy(results[mask==1].reshape(xs.shape[0], mask[0].sum()), ys[mask==1].reshape(xs.shape[0], mask[0].sum()))
            loss = loss_func(last[mask == 1].cuda(), results[mask == 1].cuda())
            acc_list.append(acc)
            cross_entropy_list.append(loss.item())
        index = torch.argmin(torch.tensor(cross_entropy_list))
    return acc_list[index], index