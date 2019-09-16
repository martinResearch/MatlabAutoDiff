rng(1)
% testing repmat
f=@(x) repmat(x,[3,2]);
CheckAutoDiffJacobian(f,rand(2,3),1e-9);

%testing compatible size multiplication (i.e. using broadcasting)
f=@(x) x.*[3,4,2];
CheckAutoDiffJacobian(f,rand(2,3),1e-9);
f=@(x) [3,4,2].*x;
CheckAutoDiffJacobian(f,rand(2,3),1e-9);
f=@(x) x(1,:).*x;
CheckAutoDiffJacobian(f,rand(2,3),1e-9);

% various tests
f=@(x) x';
CheckAutoDiffJacobian(f,randn(2,3),1e-9);

f=@(x) abs(x);
CheckAutoDiffJacobian(f,randn(2,3),1e-9);

f=@(x) sqrt(x);
CheckAutoDiffJacobian(f,rand(2,3),1e-8);

f=@(x) cos(x);
CheckAutoDiffJacobian(f,rand(2,3),1e-9);

f=@(x) sin(x);
CheckAutoDiffJacobian(f,rand(2,3),1e-9);
        
f=@(x) tan(x);
CheckAutoDiffJacobian(f,rand(2,3),1e-9);
        
f=@(x) acos(x);
CheckAutoDiffJacobian(f,rand(2,3),1e-9);
        
f=@(x) asin(x);
CheckAutoDiffJacobian(f,rand(2,3),1e-9);
        
f=@(x) atan(x);
CheckAutoDiffJacobian(f,rand(2,3),1e-9);

f=@(x) exp(x);
CheckAutoDiffJacobian(f,rand(2,3),1e-9);

f=@(x) log(x);
CheckAutoDiffJacobian(f,rand(2,3),1e-9);
        
f=@(x) tanh(x);
CheckAutoDiffJacobian(f,rand(2,3),1e-9);


f=@(x) conj(x);
CheckAutoDiffJacobian(f,rand(2,3),1e-9);

%f=@(x) cat(1,x,x*2,rand(3,3));
%CheckAutoDiffJacobian(f,rand(2,3),1e-9);

f=@(x) repmat(x,[3,4]);
CheckAutoDiffJacobian(f,rand(2,3),1e-9);      
         
f=@(x) diag(x);
CheckAutoDiffJacobian(f,rand(4,1),1e-9);
CheckAutoDiffJacobian(f,rand(4,4),1e-9); 
        
  
f=@(x) diff(x,1,2);
CheckAutoDiffJacobian(f,rand(4,3),1e-9);
f=@(x) diff(x,1,1);
CheckAutoDiffJacobian(f,rand(4,3),1e-9);
            
f=@(x) x(:,end);
CheckAutoDiffJacobian(f,rand(4,3),1e-9); 
f=@(x) x(end,:);
CheckAutoDiffJacobian(f,rand(4,3),1e-9);
 
f=@(x) x(2,:);
CheckAutoDiffJacobian(f,rand(4,3),1e-9);  

%f=@(x) max(x);
%CheckAutoDiffJacobian(f,rand(4,3),1e-9); 

f=@(x) max(x,-x);
CheckAutoDiffJacobian(f,rand(4,3),1e-9); 
 
%f=@(x) min(x);
%CheckAutoDiffJacobian(f,rand(4,3),1e-9); 

%f=@(x) min(x,-x);
%CheckAutoDiffJacobian(f,rand(4,3),1e-9); 

%f=@(x) x-x(1,2);
%CheckAutoDiffJacobian(f,rand(4,3),1e-9); 

f=@(x) x-3;
CheckAutoDiffJacobian(f,rand(4,3),1e-9); 

f=@(x) 3-x;
CheckAutoDiffJacobian(f,rand(4,3),1e-9); 

%f=@(x) x^2;
%CheckAutoDiffJacobian(f,rand(3,3),1e-9); 

f=@(x) x.^2;
CheckAutoDiffJacobian(f,rand(3,2),1e-9); 

%f=@(x) inv(x);
%CheckAutoDiffJacobian(f,rand(3,2),1e-9); 

f=@(x) x/x(2,2);
CheckAutoDiffJacobian(f,rand(3,2),1e-9); 

f=@(x) x/3;
CheckAutoDiffJacobian(f,rand(3,2),1e-9); 

%f=@(x) x./x(2,2);
%CheckAutoDiffJacobian(f,rand(3,2),1e-9); 
f=@(x) x./3;
CheckAutoDiffJacobian(f,rand(3,2),1e-9); 
%f=@(x) 3./x;
%CheckAutoDiffJacobian(f,rand(3,2),1e-9); 

f=@(x) x.*abs(x);
CheckAutoDiffJacobian(f,randn(3,3),1e-9); 

f=@(x) x.*x(2,2);
CheckAutoDiffJacobian(f,rand(3,2),1e-9); 
        
%f=@(x) norm(x);
%CheckAutoDiffJacobian(f,rand(3,2),1e-9); 

f=@(x) x+x(:,1);
CheckAutoDiffJacobian(f,rand(3,2),1e-9); 

f=@(x) reshape(x,3,2);
CheckAutoDiffJacobian(f,rand(3,2),1e-9); 

%f=@(x) sort(x,1);
%CheckAutoDiffJacobian(f,rand(3,2),1e-9);

%f=@(x) subsasgn(x(3,:)=x(1,:);
%CheckAutoDiffJacobian(f,rand(3,2),1e-9);
                
%f=@(x) x(3,:,:);
%CheckAutoDiffJacobian(f,rand(3,2,4),1e-9); 

%f=@(x) x(:,3,:);
%CheckAutoDiffJacobian(f,rand(3,2,4),1e-9); 
        
f=@(x) sum(x,2);
CheckAutoDiffJacobian(f,rand(3,2),1e-9);    
   
%f=@(x) mean(x,2);
%CheckAutoDiffJacobian(f,rand(3,2,4),1e-9); 
  
%times
%f=@(x) x.*abs(x);
%CheckAutoDiffJacobian(f,randn(3,2,4),1e-9); 
%f=@(x) x.*randn(3,2,4);
%CheckAutoDiffJacobian(f,randn(3,2,4),1e-9);  
%f=@(x) randn(3,2,4).*x;
%CheckAutoDiffJacobian(f,randn(3,2,4),1e-9); 
 

%f=@(x) eig(x);
%CheckAutoDiffJacobian(f,randn(3,3),1e-9);

f=@(x) x';
CheckAutoDiffJacobian(f,randn(3,3),1e-9);

%f=@(x) permute(x,[3,1,2]);
%CheckAutoDiffJacobian(f,randn(3,2,4),1e-9);
                
%f=@(x) x-1;
%CheckAutoDiffJacobian(f,randn(3,2,4),1e-9);             
        
%f=@(x) x+1;
%CheckAutoDiffJacobian(f,randn(3,2,4),1e-9);             
        
%f=@(x) [x,x*2];
%CheckAutoDiffJacobian(f,randn(3,2,4),1e-9);             
        
%f=@(x) [x;x*2];
%CheckAutoDiffJacobian(f,randn(3,2,4),1e-9);             
   
f=@(x) det(x);
CheckAutoDiffJacobian(f,randn(2,2),1e-9);

%f=@(x) det(x);
%CheckAutoDiffJacobian(f,randn(3,3),1e-9);  


        
       



        