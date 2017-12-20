netdef = {}

-- Free-form region based attention (MFA-R)
function netdef.MFA1g(rnn_size_q,nhimage,common_embedding_size,glimpse,dropout)
   -- [Input]: {V1,Q,V2}: {100x196x2048, 100x2400, 100x19x4097}
   -- [Output]: {Fv1: 100x2400} 
   local p = dropout or .5  -- dropout ratio
   local activation = 'Tanh'
   local glimpse=glimpse or 2

   -- {V1,Q,V2}: {100x196x2048, 100x2400, 100x19x4097}
   local attention=nn.Sequential()  -- attention networks
         :add(nn.ParallelTable()
            :add(nn.Sequential()
               :add(nn.View(-1, nhimage))
               :add(nn.Dropout(p))
               :add(nn.Linear(nhimage, common_embedding_size))
               :add(nn[activation]())
               :add(nn.View(-1, 14*14, common_embedding_size))) -- {V1': 100x196x1200}
            :add(nn.Sequential()
               :add(nn.Dropout(p))
               :add(nn.Linear(rnn_size_q, common_embedding_size))
               :add(nn[activation]())
               :add(nn.Replicate(14*14, 2))) -- {Q': 100x196x1200}
            :add(nn.Sequential() --{100x19x4097}
               :add(nn.View(-1, 4097))
               :add(nn.Dropout(p))
               :add(nn.Linear(4097, common_embedding_size)) 
               :add(nn[activation]()) --{1900x2400}
               :add(nn.View(-1, 19, common_embedding_size)) --{100x19x2400}
               :add(nn.Mean(2)) --{100x1200}
               :add(nn.Replicate(14*14, 2)))) -- {V2': 100x196x1200}

         :add(nn.CMulTable()) -- {100x196x1200}
         :add(nn.View(-1, 1200)) -- {19600x1200}
         :add(nn.Normalize(2))
         :add(nn.View(-1, 14*14, common_embedding_size))
         :add(nn.View(-1, 14, 14, common_embedding_size)) --{100x14x14x1200}
         :add(nn.Transpose({3,4},{2,3})) --{100x1200x14x14}
         :add(nn.SpatialConvolution(common_embedding_size,glimpse,1,1,1,1)) --{100x2x14x14}
         :add(nn.View(-1, glimpse, 14*14)) --{100x2x196}
         :add(nn.SplitTable(2)) --{100x196, 100x196}

   local para_softmax=nn.ParallelTable()
   for j=1,glimpse do
      para_softmax:add(nn.SoftMax())
   end
   attention:add(para_softmax)  
   -- {a1,a2}: {100x196, 100x196}
   
   -- {V1,Q,V2,a1,a2}: {100x196x2048, 100x2400, 100x19x4097, 100x196, 100x196}
   local glimpses=nn.ConcatTable()
   for i=1,glimpse do
      local visual_embedding_=nn.Sequential()
            :add(nn.ConcatTable()
               :add(nn.SelectTable(3+i))   -- softmax [3~]
               :add(nn.SelectTable(1)))  -- v
            :add(nn.ParallelTable()
               :add(nn.Identity()) -- {a1: 100x196}
               :add(nn.SplitTable(2))) -- {v1-v196: 100x2048, 100x2048, ..., 100x2048}
            :add(nn.MixtureTable()) -- {100x2048}
            :add(nn.Dropout(p))
            :add(nn.Linear(nhimage, common_embedding_size))
            :add(nn[activation]())
      glimpses:add(visual_embedding_) -- {100x1200}
   end

   local visual_embedding=nn.Sequential()
         :add(glimpses) -- {100x1200, 100x1200}
         :add(nn.JoinTable(2)) 
   -- {V1': 100x2400}
   
   -- {V1: 100x2048x14x14}
   local reshaper
   reshaper = nn.Sequential()
      :add(nn.Transpose({2,3},{3,4}))
      :add(nn.Reshape(14*14, nhimage))
   -- {100x196x2048}

   -- {V1,Q,V2}: {100x2048x14x14, 100x2400, 100x19x4097}
   local multimodal_net=nn.Sequential()
      :add(nn.ParallelTable()
         :add(reshaper)     -- {100x196x2048}
         :add(nn.Identity()) -- {100x2400}
         :add(nn.Identity())) -- {100x19x4097}
 
      :add(nn.ConcatTable()
         :add(nn.SelectTable(1))  -- v1
         :add(nn.SelectTable(2))  -- q
         :add(nn.SelectTable(3))  -- v2
         :add(attention))  
       -- {100x196x2048, 100x2400, 100x19x4097, {100x196, 100x196}}

      :add(nn.FlattenTable()) -- {100x196x2048, 100x2400, 100x19x4097, 100x196, 100x196}

      :add(nn.ConcatTable()
         :add(visual_embedding)  -- {V1':100x2400}
         :add(nn.Sequential()
            :add(nn.SelectTable(2)) -- {Q:100x2400}
            :add(nn.Dropout(p))
            :add(nn.Linear(rnn_size_q, common_embedding_size*glimpse,dropout)) -- {100x2400}
            :add(nn[activation]())) -- {Q': 100x2400}
      )-- {V1',Q'}: {100x2400, 100x2400}

      :add(nn.CMulTable())
      -- {Fv1: 100x2400}   

   return multimodal_net
end


-- Detection based attention (MFA-D)
function netdef.MFA2g(rnn_size_q,nhimage,common_embedding_size,glimpse,dropout)
   -- [Input]: {V1,Q,V2}: {100x196x2048, 100x2400, 100x19x4097}
   -- [Output]: {Fv1: 100x2400} 
   local p = dropout or .5  -- dropout ratio
   local activation = 'Tanh'

   local rnn_size_q=rnn_size_q or 1200
   local nhimage = 4097
   local common_embedding_size=common_embedding_size or 1200
   local glimpse=glimpse or 2

   -- {V1,Q,V2}: {100x196x2048, 100x2400, 100x19x4097}
   local attention=nn.Sequential()  -- attention networks
         :add(nn.ParallelTable()
             :add(nn.Sequential() --{V1:100x196x2048}
               :add(nn.View(-1, 2048))
               :add(nn.Dropout(p))
               :add(nn.Linear(2048, common_embedding_size)) 
               :add(nn[activation]()) --{1900x1200}
               :add(nn.View(-1, 14*14, common_embedding_size)) --{100x196x1200}
               :add(nn.Mean(2)) --{100x1200}
               :add(nn.Replicate(19, 2))) -- {V1': 100x19x1200}
            :add(nn.Sequential() -- {Q: 100x2400}
               :add(nn.Dropout(p))
               :add(nn.Linear(rnn_size_q, common_embedding_size))
               :add(nn[activation]())
               :add(nn.Replicate(19*1, 2))) -- {Q': 100x19x1200}
            :add(nn.Sequential() -- {V2: 100x19x4097}
               :add(nn.View(-1, nhimage))
               :add(nn.Dropout(p))
               :add(nn.Linear(nhimage, common_embedding_size))
               :add(nn[activation]())
               :add(nn.View(-1, 19*1, common_embedding_size)))) -- {V2': 100x19x1200}
         
         :add(nn.CMulTable()) --{100x19x1200}
         :add(nn.View(-1, common_embedding_size)) --{1900x1200}
         :add(nn.Normalize(2))

         :add(nn.View(-1, 19, 1, common_embedding_size)) --{100x19x1x1200}
         :add(nn.Transpose({3,4},{2,3}))   --{100x1200x19x1}     
         :add(nn.SpatialConvolution(common_embedding_size,glimpse,1,1,1,1))
         :add(nn.View(-1, glimpse, 19*1)) --{100x2x19}
         :add(nn.SplitTable(2)) --{{100x19}, {100x19}}

   local para_softmax=nn.ParallelTable()
   for j=1,glimpse do
      para_softmax:add(nn.SoftMax())
   end
   attention:add(para_softmax)  
   --[Attention Weight, a1,a2]: {{100x19}, {100x19}}
   
   -- {V1,Q,V2,a1,a2}: {100x196x2048, 100x2400, 100x19x4097, 100x19, 100x19}
   local glimpses=nn.ConcatTable()
   for i=1,glimpse do
      local visual_embedding_=nn.Sequential()
            :add(nn.ConcatTable()
               :add(nn.SelectTable(3+i))   -- softmax [4~]
               :add(nn.SelectTable(3)))  -- v2
            :add(nn.ParallelTable()
               :add(nn.Identity()) -- {a1: 100x19}
               :add(nn.SplitTable(2))) -- {v1-v19: 100x4097, 100x4097, ..., 100x4097}
            :add(nn.MixtureTable()) -- {100x4097}
            :add(nn.Dropout(p))
            :add(nn.Linear(nhimage, common_embedding_size))
            :add(nn[activation]())
      glimpses:add(visual_embedding_) -- {100x1200}
   end

   local visual_embedding=nn.Sequential()
         :add(glimpses) -- {100x1200, 100x1200}
         :add(nn.JoinTable(2))  
   -- [Attended Visual Feature]: {V2': 100x2400}

   -- {V1: 100x2048x14x14}
   local reshaper
   reshaper = nn.Sequential()
      :add(nn.Transpose({2,3},{3,4}))
      :add(nn.Reshape(14*14, 2048))
   -- {100x196x2048}

   -- {V1,Q,V2}: {100x2048x14x14, 100x2400, 100x19x4097}
   local multimodal_net=nn.Sequential()
      :add(nn.ParallelTable()
         :add(reshaper)     -- {100x196x2048}
         :add(nn.Identity()) -- {100x2400}
         :add(nn.Identity())) -- {100x19x4097}

      :add(nn.ConcatTable()
         :add(nn.SelectTable(1))  -- v1
         :add(nn.SelectTable(2))  -- q
         :add(nn.SelectTable(3))  -- v2
         :add(attention))
      -- {100x196x2048, 100x2400, 100x19x4097, {100x19, 100x19}}

      :add(nn.FlattenTable()) -- {100x196x2048, 100x2400, 100x19x4097, 100x19, 100x19}

      :add(nn.ConcatTable()
         :add(visual_embedding)  -- {V2': 100x2400} 
         :add(nn.Sequential()
            :add(nn.SelectTable(2)) -- {Q:100x2400} 
            :add(nn.Dropout(p))
            :add(nn.Linear(rnn_size_q, common_embedding_size*glimpse,dropout)) -- {100x2400}
            :add(nn[activation]())) -- {Q': 100x2400} 
      )-- {V2',Q'}: {100x2400, 100x2400}

      :add(nn.CMulTable()) -- {100x2400}
      -- {Fv2: 100x2400}   
   
   return multimodal_net
end