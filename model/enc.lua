local enc = {}

function enc.mlp(size, output_size, num_layers, f)
    -- size = dimensionality of inputs
    -- num_layers = number of hidden layers (default = 1)
    -- f = non-linearity (default = ReLU)
    
    local output
    local num_layers = num_layers or 1
    local f = f or nn.ReLU()
    local input = nn.Identity()()
    local inputs = {[1]=input}
    for i = 1, num_layers-1 do        
        output = f(nn.Linear(size, size)(inputs[i]))
	table.insert(inputs, output)
    end
    output = f(nn.Linear(size, output_size)(inputs[num_layers]))
    
    return nn.gModule({input},{output})
end

return enc
