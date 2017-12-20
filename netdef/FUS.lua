netdef = {}

function netdef.CAT(common_embedding_size,noutput,glimpse,dropout)
   local p = dropout or .5  -- dropout ratio
   local activation = 'Tanh'

   local fusion_net = nn.Sequential()

   fusion_net:add(nn.Concat(2)
      :add(nn.SelectTable(1))   -- F1
      :add(nn.SelectTable(2)))  -- F2  --> F1|F2

   fusion_net:add(nn.Linear(common_embedding_size*glimpse*2, noutput))

   return fusion_net
end

function netdef.ADD(common_embedding_size,noutput,glimpse,dropout)
   local p = dropout or .5  -- dropout ratio
   local activation = 'Tanh'

   local fusion_net = nn.Sequential()

   fusion_net:add(nn.CAddTable())

   fusion_net:add(nn.Dropout(p))
             :add(nn.Linear(common_embedding_size*glimpse, noutput))

   return fusion_net
end

function netdef.MUL(common_embedding_size,noutput,glimpse,dropout)
   local p = dropout or .5  -- dropout ratio
   local activation = 'Tanh'

   local fusion_net = nn.Sequential()

   fusion_net:add(nn.CMulTable())

   fusion_net:add(nn.Linear(common_embedding_size*glimpse, noutput))

   return fusion_net
end