import torch
import torch.nn as nn
import numpy as np
import math

from emb_utils import get_cum_prod


class quantize(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input, scale, bit=8):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """

        

        max_q = 2**(bit-1)-1
        min_q = -2**(bit-1)

        quant = lambda x: torch.clamp(torch.round(x), min = min_q, max = max_q)
     
        ctx.save_for_backward(input, scale)
        ctx.quant = quant
        ctx.input_div_scale = input/scale
        ctx.q_input = quant(ctx.input_div_scale)
        ctx.min_q = torch.tensor(min_q)
        ctx.max_q = torch.tensor(max_q)

        return scale * ctx.q_input
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, scale= ctx.saved_tensors
        grad_input = grad_output*torch.where((ctx.input_div_scale<=ctx.max_q) & (ctx.input_div_scale>=ctx.min_q), 1.0, 0.0)
        
        grad_scale = (torch.where((ctx.input_div_scale<=ctx.max_q) & (ctx.input_div_scale>=ctx.min_q), ctx.q_input - ctx.input_div_scale, ctx.input_div_scale))


        grad_scale = grad_output*torch.clamp(grad_scale, min = ctx.min_q.to(grad_scale.device), max = ctx.max_q.to(grad_scale.device))


        return grad_input, grad_scale, None

class Linear_TT(nn.Module):
    def __init__(self,
                in_features,
                out_features,
                TT_dims=None,
                TT_ranks=None,
                bias=True
    ):
        super(Linear_TT, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.TT_dims = TT_dims
        self.TT_ranks = TT_ranks
        
        target_stddev = np.sqrt(1/(self.in_features+self.out_features))
        
        N = len(self.TT_dims)
        
        self.TT_cores = nn.ParameterList()
        for i in range(N):
            r1 = self.TT_ranks[i]
            r2 = self.TT_ranks[i+1]
            U = nn.Parameter(torch.randn(r1, self.TT_dims[i], r2)/np.sqrt(r2)*target_stddev**(1/N))
            self.TT_cores.append(U)
            
        if bias:
            stdv = 1. / np.sqrt(out_features)

            self.register_parameter('bias', nn.Parameter(torch.randn(out_features)))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.bias = None
            
        self.quantization_aware = False
            
    def init_scale(self,bit_cores=8,bit_intermediate=8):
        self.bit_cores = bit_cores
        self.bit_intermediate = bit_intermediate
        
        self.scale_other = torch.nn.ParameterList()
        self.scale_cores = torch.nn.ParameterList()
        
        max_q = 2**(bit_cores-1)-1


        for U in self.TT_cores:
            scale_core = U.data.abs().max()/max_q
            self.scale_cores.append(torch.nn.Parameter(torch.tensor(scale_core)))

        
        max_q_intermediate = 2**(bit_intermediate-1)-1
        scale_input = 1/max_q
        scale_intermediate = 1/max_q_intermediate
        scale_x = 1/max_q_intermediate

        
        
        self.scale_input = torch.nn.Parameter(torch.tensor(scale_input))
        self.scale_intermediate = torch.nn.Parameter(torch.tensor(scale_intermediate))
        self.scale_x = torch.nn.Parameter(torch.tensor(scale_x))



        self.scale_other.append(self.scale_input)
        self.scale_other.append(self.scale_intermediate)
        self.scale_other.append(self.scale_x)

            
    def forward(self,input):

        factors = self.TT_cores
        

        if self.quantization_aware==False:
            out = self.forward_tt_full_precision(input,factors)  
            

        elif self.quantization_aware==True:
            out = self.forward_tt_quantization_aware(input,factors) 

        if self.bias is not None:
            out = out + self.bias


        return out 
    

    def forward_tt_quantization_aware(self,input,factors):

        input = quantize.apply(input,self.scale_input,self.bit_cores)
        Q_factors = []
        for i,U in enumerate(factors):
            Q_factors.append(quantize.apply(U,self.scale_cores[i],self.bit_cores))
        factors = Q_factors

        quant_intermediate = lambda x: quantize.apply(x,self.scale_intermediate,self.bit_intermediate)

        quant_x = lambda x: quantize.apply(x,self.scale_x,self.bit_intermediate)


        m = len(factors)//2
        N = len(input.shape)
        if len(input.shape)==2:
            mat_shape = [input.shape[0]] + [U.shape[1] for U in factors[0:m]]
        elif len(input.shape)==3:
            mat_shape = [input.shape[0]]+[input.shape[1]] + [U.shape[1] for U in factors[0:m]]

        input = torch.reshape(input, [1] + mat_shape)
        

      
        out = factors[0]
        
        out = torch.squeeze(out)

        for i in range(1,m):
            U = factors[i]
            out = quant_intermediate(torch.tensordot(out, U, [[-1],[0]]))


        # S = 100
        out = quant_x(torch.tensordot(input, out, [list(range(N,N+m)), list(range(0,m))]))

        out = [out] + list(factors[m:])



        N = len(out[0].shape)
        output = factors[m]


        for i in range(m+1,2*m):
            U = factors[i]
            output = quant_intermediate(torch.tensordot(output,U,[[-1],[0]]))
        
        output = torch.tensordot(out[0],output,[[-1],[0]])
        # output = quant_out(output)

        output = torch.flatten(output, start_dim = N-1, end_dim = -1)
        output = torch.squeeze(output)


        return output

    
    def forward_tt_full_precision(self,input_mat,factors):
       

        
        m = len(factors)//2
        N = len(input_mat.shape)
        
        input_mat = torch.reshape(input_mat,[1]+list(input_mat.shape[0:N-1])+self.TT_dims[:m])

            
            

      
        out = factors[0]
        
        out = torch.squeeze(out)
        output = factors[m]

        for i in range(1,m):
            U = factors[i]
            V = factors[i+m]
            
            out = torch.tensordot(out, U, [[-1],[0]])
            output = torch.tensordot(output,V,[[-1],[0]])

        
        
        out = torch.tensordot(input_mat, out, [list(range(N,N+m)), list(range(0,m))]) 
    

        N = len(out.shape)
        
        
        output = torch.tensordot(out,output,[[-1],[0]])

        output = torch.flatten(output, start_dim = N-1, end_dim = -1)
        output = torch.squeeze(output)


        return output   
            
            
class Embedding_TTM_order4(nn.Module):
    '''
        optized TTM format embedding table with TTM tensors with order 4.
    '''
    def __init__(self,
                in_features,
                out_features,
                TTM_dims=None,
                TTM_ranks=None):

        super(Embedding_TTM_order4,self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.shape = TTM_dims

        # target_stddev = np.sqrt(1/(np.prod(self.shape[0])+np.prod(self.shape[1])))
        target_stddev = 1.0

        self.TTM_cores = nn.ParameterList()
        
        self.init_TTM(TTM_dims,TTM_ranks,target_stddev)
        
        
        self.ind2coord = self.get_indices_dict().to('cuda')
        
        self.register_buffer('dict_tensor',torch.zeros(in_features,dtype=torch.long))
        self.register_buffer('dict_range',torch.arange(0,in_features,dtype=torch.long))
        
        self.cum_prod = get_cum_prod(self.shape)
        
        
        self.quantization_aware = False
        
    def init_TTM(self,TTM_dims,TTM_ranks,target_sdv=1.0):
        shape = TTM_dims
        ranks = TTM_ranks
        order = len(shape[0])

        
        for i in range(order):
            n1 = shape[0][i]
            n2 = shape[1][i]
            r1 = ranks[i]
            r2 = ranks[i+1]
            U = torch.nn.Parameter(torch.randn(r1,n1,n2,r2)/math.sqrt(r2)*target_sdv**(1/order))
            self.TTM_cores.append(U)

        
    def get_indices_dict(self):
        ind2coord = []
        shape = [self.shape[0][0]*self.shape[0][1],self.shape[0][2]*self.shape[0][3]]

        for i in range(self.in_features):
            coords = np.unravel_index(i,shape)
            ind2coord.append(coords)
        ind2coord = torch.tensor(ind2coord)
        return ind2coord

    def TTM_select_ind(self,factors,inds,ind2coord_dict,use_unique=False):
        input_shape = [U.shape[1] for U in factors]
        output_shape = [U.shape[2] for U in factors]

        ranks = [U.shape[-1] for U in factors]


        out1 = torch.tensordot(factors[0],factors[1],[[-1],[0]]).movedim([3],[2]).reshape(input_shape[0]*input_shape[1],output_shape[0]*output_shape[1],ranks[1])
        out2 = torch.tensordot(factors[2],factors[3],[[-1],[0]]).movedim([3],[2]).reshape(ranks[1],input_shape[2]*input_shape[3],output_shape[2]*output_shape[3])

        if use_unique:        
            inds_cpu = inds
            inds_unique = torch.unique(inds_cpu)
            self.dict_tensor[inds_unique] = self.dict_range[0:inds_unique.shape[0]]
            
            targets = ind2coord_dict[inds_unique,:]

            out1 = out1[targets[:,0],:,:]
            out2 = out2[:,targets[:,1],:]

            out = torch.einsum('abc,cad->abd',out1,out2).flatten(start_dim=1)
            out = out[self.dict_tensor[inds_cpu],:]
        else:
            inds_cpu = inds
            targets = ind2coord_dict[inds_cpu,:]

            out1 = out1[targets[:,0],:,:]
            out2 = out2[:,targets[:,1],:]
            out = torch.einsum('abc,cad->abd',out1,out2).flatten(start_dim=1)

        return out

    def set_scale_factors(self,bit=8):
        
        self.bit = bit
        self.scale_cores = torch.nn.ParameterList()

        max_q = 2**(bit-1)-1
        for U in self.TTM_cores:
            scale_core = U.data.abs().max()/max_q
            self.scale_cores.append(torch.nn.Parameter(torch.tensor(scale_core,requires_grad=True)))
    



    def forward(self, x, use_unique=False):

        xshape = list(x.shape)
        xshape_new = xshape + [self.out_features, ]

        x = torch.flatten(x)

    
        factors = self.TTM_cores
          
        if self.quantization_aware==False:
            rows = self.TTM_select_ind(factors,x,self.ind2coord,use_unique=use_unique)
        else:
            Q_factors = []
            for i,U in enumerate(factors):
                Q_factors.append(quantize.apply(U,self.scale_cores[i],self.bit)) 
            factors = Q_factors
            rows = self.TTM_select_ind(factors,x,self.ind2coord,use_unique=use_unique)
    

        rows = rows.view(*xshape_new)


        return rows
    
    
    
def set_quantization_aware_TT(layer,bit_cores=8,bit_intermediate=8):
    layer.quantization_aware = True
    layer.init_scale(bit_cores,bit_intermediate)


    
def set_quantization_aware_TTM(layer,bit_cores=8):
    layer.quantization_aware = True
    layer.set_scale_factors(bit_cores)

def set_quantization_aware_model(model,bit_cores=8,bit_intermediate=8):
    for n,p in model.bert.named_modules():
        if type(p).__name__ == 'Linear_TT':
            set_quantization_aware_TT(p,bit_cores,bit_intermediate)
        elif type(p).__name__ == 'Embedding_TTM_order4':
            set_quantization_aware_TTM(p,bit_cores)
    
    
def Get_tensor_TT(model,TT_dims_att,TT_ranks_att,TT_dims_ffn,TT_ranks_ffn):
    for n,p in model.bert.named_modules():
        if type(p).__name__ == 'Linear':
            print(f"processing {n}")
            if 'attention' in n:
                TT_dims = TT_dims_att
                TT_ranks = TT_ranks_att
            elif 'intermediate' in n:
                TT_dims = TT_dims_ffn
                TT_ranks = TT_ranks_ffn
            elif 'pooler' in n:
                TT_dims = TT_dims_att
                TT_ranks = TT_ranks_att
            elif 'output' in n and 'dense' in n:
                TT_dims = TT_dims_ffn[::-1]
                TT_ranks = TT_ranks_ffn[::-1]
            
            key_previous = '.'.join(n.split('.')[:-1])
            mod = model.bert.get_submodule(key_previous)

            W = p.weight
            out_features,in_features = p.weight.shape
            
            Layer = Linear_TT(in_features,out_features,TT_dims,TT_ranks,bias=(p.bias!=None))
            Layer.to(W.device).to(W.dtype)

            if p.bias!=None:
                Layer.bias.data = p.bias
            else:
                Layer.bias = None
            

           
            
            del p.weight
            del p
            
            print(key_previous,n.split('.')[-1])
            setattr(mod, n.split('.')[-1], Layer)
            

            
            
def get_tensor_model(model,TT_dims_att,TT_ranks_att,TT_dims_ffn,TT_ranks_ffn,TTM_dims,TTM_ranks):
    Get_tensor_TT(model,TT_dims_att,TT_ranks_att,TT_dims_ffn,TT_ranks_ffn)
    
    # emb = model.bert.embeddings.word_embeddings
    # weight = emb.weight
    # in_features,out_features = weight.shape
    
    # Layer = Embedding_TTM_order4(in_features,out_features,TTM_dims,TTM_ranks)
    # Layer.to(weight.device).to(weight.dtype)
    
    # del emb.weight
    # del emb
    
    # setattr(model.bert.embeddings, 'word_embeddings', Layer)
    
    
