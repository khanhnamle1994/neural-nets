function ret = cd1(rbm_w, visible_data)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_data> is a (possibly but not necessarily binary) matrix of size <number of visible units> by <number of data cases>
% The returned value is the gradient approximation produced by CD-1. It's of the same shape as <rbm_w>.
    visible_data = sample_bernoulli(visible_data);

    h_p = visible_state_to_hidden_probabilities(rbm_w, visible_data); % size <number of hidden units> by <number of configurations that we're handling in parallel>.
    h_ = sample_bernoulli(h_p);

    d_1 = configuration_goodness_gradient(visible_data, h_);

    % reconstruction
    v_p = hidden_state_to_visible_probabilities(rbm_w, h_); %  size <number of visible units> by <number of configurations that we're handling in parallel>.
    v_ = sample_bernoulli(v_p);

    h_p = visible_state_to_hidden_probabilities(rbm_w, v_); % size <number of hidden units> by <number of configurations that we're handling in parallel>.
    h_ = sample_bernoulli(h_p); % Instead of a sampled state, we'll simply use the conditional probabilities.

    d_2 = configuration_goodness_gradient(v_, h_p); % use h_p instead of h_, reason see README.md, question 8

    ret = d_1 - d_2;
end
