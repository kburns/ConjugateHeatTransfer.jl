

using LinearAlgebra


function polyfitA(x,ZCs,NB,f,n)
    numbodies=length(x)
    mtot=sum(length.(x));
    RHSvec=Complex.(zeros(mtot,1));
    mrun=0
    stackedx=x[1]
    for kk=1:length(x); #number of bodies
        lencurr=length(x[kk])
        RHSvec[mrun+1:mrun+lencurr]=f[kk]
        mrun=mrun+lencurr;
        if kk<numbodies
            stackedx=[stackedx;x[kk+1]]#don't exceed the index.
        end
    end

    Qnet = Complex.(zeros(mtot,2*numbodies*n+numbodies+1));
    dout = Complex.(zeros((numbodies*(n+1)+1),1));
    HH=Matrix{ComplexF64}[];
    mt=0;
    for ii = 1:length(ZCs)
        m=length(stackedx)#number of points associated with ii'th body, currently under consideration.
        Q = Complex.(ones(m,n+1));
        H = Complex.(zeros(n+1,n));
        for k = 1:n
            q = ((stackedx .- ZCs[ii]).^(-1)).*Q[:,k];
            for j = 1:k
                H[j,k] = Q[:,j]'*q/m;
                q = q - H[j,k]*Q[:,j];
            end
            H[k+1,k] = norm(q)/sqrt(m);
            Q[:,k+1] = q/H[k+1,k];
        end
        push!(HH,H);
        #HH[ii]=H[:,:]; #copy the matrix to the first entry of HH.
        Qnet[:,1+ (2*(ii-1)*n+1):1+(2*(ii-1)*n+n)]=real(Q[:,2:n+1]);
        Qnet[:,1+n+ (2*(ii-1)*n+1):(1+n+2*(ii-1)*n+n)]=imag(Q[:,2:n+1]);
        Qnet[mt+1:mt+length(x[ii]),numbodies*2*n+1+ii]=Complex.(ones(length(x[ii]),1));
        mt = mt+length(x[ii]);#running total to put the "ones" constants in the correct spot.
    end
    Qnet[:,1]=Complex.(ones(mtot,1));

    #structure of Qnet
    #Qnet = [1 Re(Series1) Im(Series1) Re(Series2) Im(Series2)]

    d = Qnet\RHSvec;
    println("Fitting error: ", norm(Qnet*d-RHSvec, Inf))

    dout[1]=im*d[1];
    for ii=1:numbodies
        dout[(ii-1)*n+2:ii*n+1] = im*d[2*(ii-1)*n+2:2*(ii-1)*n+n+1] .+ d[2*(ii-1)*n+n+2:2*(ii-1)*n+2*n+1];
        #dout[2*n+1+ii] = im*d[4*n+1+ii];
    end
    # dout[2*n+2] = im*d[4*n+2];
    # dout[2*n+3] = im*d[4*n+3];
    return dout, HH;
end


function polyvalA(d,HH,ZCs,s)
    numbodies=length(HH)
    n = size(HH[1],2);
    y=0
    #W = Complex.(ones(M,n+1));
    for ii= 1:numbodies
        W = Complex.(ones(n+1));
        H=HH[ii]
        for k = 1:n
            w = ((s .- ZCs[ii]).^(-1)).*W[k];
            #w = s.*W[:,k];
            for j = 1:k
                w = w - H[j,k]*W[j];
            end
            #W[:,k+1] = w/H[k+1,k];
            W[k+1]=w/H[k+1,k];
        end
        y = y + W[2:n+1]'*d[n*(ii-1)+2:n*(ii-1)+n+1];
    end
    y = y + d[1]; #remember to add back the constant
    return y;
end


function solve_potential_flow(bodies, U, N_samp, NL) #NL =Laurent series length
    samp_points = Vector(range(0,2*pi,length=N_samp))# maybe elim endpoints... later
    bod = [b.zθ.(samp_points) for b in bodies]
    ZCs = [b.zc for b in bodies]
    dout,hh = polyfitA(bod,ZCs,N_samp,-imag(conj(U)*bod),NL)
    W(z) = polyvalA(dout,hh,ZCs, z) + conj(U) * z
    return W
end