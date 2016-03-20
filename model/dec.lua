local dec = {}

function dec.mlp(size, output_size, num_layers, f, dropout)
    -- size = dimensionality of inputs
    -- num_layers = number of hidden layers (default = 1)
    -- f = non-linearity (default = ReLU)
    
    local output
    local num_layers = num_layers or 1
    local f = f or nn.ReLU()
    local dropout = dropout or 0.3
    local input = nn.Identity()()
    local inputs = {[1]=input}
    for i = 1, num_layers-1 do        
        output = f(nn.Linear(size, size)(inputs[i]))
	table.insert(inputs, output)
    end
    word_vec_layer = nn.Linear(size, output_size)
    word_vec_layer.name = 'word_vec'
    output = f(word_vec_layer(inputs[num_layers]))
    output = nn.LogSoftMax()(output)    
    return nn.gModule({input},{output})
end

return dec
