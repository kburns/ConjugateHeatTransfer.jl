
########################################################
############## START OF FUNCTION SECTION ###############
#1) Polyfit: Fitting for the coefficients of the series solution.
function polyfitA(x,ZCs,NB,f,n)
    numbodies=length(x)
    mtot=sum(length.(x)); 
    RHSvec=Complex.(zeros(mtot,1));
    mrun=0
    for kk=1:length(x); #number of bodies
        lencurr=length(x[kk])
        RHSvec[mrun+1:mrun+lencurr]=f[kk]
        mrun=mrun+lencurr;
    end

    Qnet = Complex.(zeros(mtot,2*numbodies*n+1));
    dout = Complex.(zeros((numbodies*n+1),1));
    println(length(dout))
    HH=Matrix{ComplexF64}[];
    mt=0;
    for ii = 1:length(ZCs)
        m=length(x[ii])#number of points associated with ii'th body, currently under consideration.
        Q = Complex.(ones(m,n+1));
        H = Complex.(zeros(n+1,n));
        for k = 1:n
            q = ((x[ii] .- ZCs[ii]).^(-1)).*Q[:,k];
            for j = 1:k
                H[j,k] = Q[:,j]'*q/m;
                q = q - H[j,k]*Q[:,j];
            end
            H[k+1,k] = norm(q)/sqrt(m);
            Q[:,k+1] = q/H[k+1,k];
        end
        push!(HH,H);
        #HH[ii]=H[:,:]; #copy the matrix to the first entry of HH.
        Qnet[mt+1:mt+m,1+ (2*(ii-1)*n+1):1+(2*(ii-1)*n+n)]=real(Q[:,2:n+1]);
        Qnet[mt+1:mt+m,1+n+ (2*(ii-1)*n+1):(1+n+2*(ii-1)*n+n)]=imag(Q[:,2:n+1]);
        mt = mt+m;#running total
    end
    Qnet[:,1]=Complex.(ones(mtot,1));
    #structure of Qnet
    #Qnet = [1 Re(Series1) Im(Series1) Re(Series2) Im(Series2)]

    d = Qnet\RHSvec;

    dout[1]=im*d[1];
    for ii=1:numbodies
        dout[(ii-1)*n+2:ii*n+1] = im*d[2*(ii-1)*n+2:2*(ii-1)*n+n+1] .+ d[2*(ii-1)*n+n+2:2*(ii-1)*n+2*n+1];
    end

    return dout, HH;
end
#2) Polyval: Evaluation of the solution is non-trivial due to the orthogonalization:
    #This function allows us to evaluate the resulting function from Polyfit in (3)
function polyvalA(d,HH,s)
    numbodies=length(HH)
    M = length(s);
    n = size(HH[1],2);
    y=Complex.(zeros(M,1));
    #W = Complex.(ones(M,n+1));
    for ii= 1:numbodies
        W = Complex.(ones(M,n+1));
        H=HH[ii]
        for k = 1:n
            w = ((s .- ZCs[ii]).^(-1)).*W[:,k];
            #w = s.*W[:,k];
            for j = 1:k
                w = w - H[j,k]*W[:,j];
            end
            #W[:,k+1] = w/H[k+1,k];
            W[:,k+1]=w/H[k+1,k];
        end
        y = y + W[:,2:n+1]*d[n*(ii-1)+2:n*(ii-1)+n+1];
    end 
    y = y .+ d[1]; #remember to add back the constant
    return y;
end
############### END OF FUNCTION SECTION ################
########################################################

function solve_pot_flow(bodies, U, N_samp, NL) #NL =Laurent series length
    samp_points = Vector(range(0,2*pi,length=N_samp))# maybe elim endpoints... later
    bod = [b.zθ(samp_points) for b in bodies]

    #Here the code starts: the only input is "bod", a list of sampled bodies, from the user.
    numbodies=length(bod);

    xv=real(bod);
    yv=imag(bod);

    #plot(xv[1],yv[1]) #Nice -- a scrunched up circle!
    ########### END OF PREDEFINED PARAMETERS ############
    #####################################################





    ######################################################
    #################### MATH PROBLEM ###################
    #NOTE: FOR THE LAURENT SERIES WE HAVE TO BE CAREFUL TO INPUT 1./Z_boundary into polyfit!
    #constadd=Complex.(zeros(2*NB,1));
    #constadd[NB+1:2*NB]=0*Complex.(ones(NB,1));
    #dout,hh = polyfitA(bod,ZCs,NB,-imag(conj(U)*bod)+constadd,NL) #the "f" RHS is OLD VERSION WITH CONSTADD
    dout,hh = polyfitA(bod,ZCs,NB,-imag(conj(U)*bod),NL) #the "f" RHS is 

    W(z) = polyval(dout,hh, z) + conj(U) * z
    return W
end