%version 2022/04/20, coded by Weiwei Cai, Hoon Hahn Yoon, and Fedor Nigmatulin
line-ending-selector:convert-to-CRLF
clear;clc;close all;

%**************parameter settings*************
NumWavelengths=19;          %Number of Wavelengths for reconstruction
NumWavelengthsInterp=NumWavelengths*19;   %number of interpolated wavelengths
NumGaussianBasis=19;        %Number of Gaussian Basis
MinLambda=400;              %smallest wavelength
MaxLambda=850;              %largest wavelength
FWHMset=100;                %arbitary

%**************loading data*************
ResponseFilename=load('LearningMatrix.txt');                      %Responsivity as function of wavelength at different Vg (Detectors) & (GateVoltage,Lambda)=(82,19)
ResponseCurves=ResponseFilename/max(max(ResponseFilename));       %normalize the input response curve

% % input unknown measured data
SignalFilename='MeausuredSignal.txt';

MeasuredSignals=load(SignalFilename);                              %Photocurrent as function of Vg (Detectors)

NumRow=size(ResponseCurves,1);                                     %Number of Vg (Detectors)
NumCol=size(MeasuredSignals,2);                                    %Multiple data processing from matrix of MeasuredSignals
SimulatedSignals=zeros(NumRow,NumCol);                             %Initialization of simulated signals
HiResReconstructedSpectrum=zeros(NumWavelengthsInterp,NumCol);     %Initialization of reconstructed spectrum for plotting
    
VecOfLambdas=linspace(MinLambda,MaxLambda,NumWavelengths)'/MaxLambda;             %normalized wavelengths
VecOfLambdasPlot=linspace(MinLambda,MaxLambda,NumWavelengthsInterp)'/MaxLambda;   %normalized interpolated wavelengths
Lambdas=linspace(MinLambda,MaxLambda,NumWavelengthsInterp)';                      %interpolated wavelengths

HiResResponseCurves=zeros(NumRow,NumWavelengthsInterp);
for r=1:NumRow
    HiResResponseCurves(r,:)=spline(VecOfLambdas,ResponseCurves(r,:),VecOfLambdasPlot);  %spline interpolation
end
    
    MinFWHM=FWHMset;                                   %unit:0.001, so it is 0.010
    MaxFWHM=FWHMset+1;                                 %unit:0.001, so it is 0.010
    FWHM=zeros(MaxFWHM-MinFWHM+1,NumCol);            
    FWHMtemp=linspace(MinFWHM,MaxFWHM,MaxFWHM-MinFWHM+1)/1000;      %all the tested FWHM's
    MeasuResidual=zeros(MaxFWHM-MinFWHM+1,NumCol);                  %record 2-norm of signal residual for all FWHMs's
    ReguTerm=zeros(MaxFWHM-MinFWHM+1,NumCol);                       %record 2-norm of regularization term for all FWHMs's
    OptimalFWHM=zeros(1,NumCol);

for i=1:NumCol

    MeasuredSignals(:,i)=MeasuredSignals(:,i)/max(max(MeasuredSignals(:,i)));   %normalized measured signals

    %**************generate Gaussian Basis here
    GaussianCenter=linspace(MinLambda,MaxLambda,NumGaussianBasis)'/MaxLambda;   %normalized wavelength centers
    GaussianBasis=zeros(NumWavelengths,NumGaussianBasis);                       %all the Gaussian basis for reconstruction
    HiResGaussianBasis=zeros(NumWavelengthsInterp,NumGaussianBasis);            %interpolated Gaussian basis 
        
    for k=1:MaxFWHM-MinFWHM+1

        FWHM(k,i)=FWHMtemp(k);
        GaussianSigma=FWHMtemp(k)/(2*sqrt(2*log(2)));        %standard deviation of the Gaussian basis, nomalized

            for j=1:NumGaussianBasis                           %generate all the Guassian basis
                HiResGaussianBasis(:,j)=exp(-0.5*((VecOfLambdasPlot-GaussianCenter(j))/GaussianSigma).^2);
                HiResGaussianBasis(:,j)=1/GaussianSigma/sqrt(2*pi)*HiResGaussianBasis(:,j);
            end

        %**************reconstruction with matlab function 'lsqnonneg'
        LaplacianMatrix=eye(NumGaussianBasis);                  %generate Laplacian matrix, which is an identity matrix
        WeightMatrix=HiResResponseCurves*HiResGaussianBasis;    %this code is equivalent to Eq. (S6) in Science paper

        OptimalGamma=FindOptGamma_GCV(MeasuredSignals(:,i),WeightMatrix);       %find optimal gamma here with GCV method

        AugWeightMatrix=[WeightMatrix;OptimalGamma^2*LaplacianMatrix];          %generate augumented matrix
        AugMeasuredSignals=[MeasuredSignals(:,i);zeros(NumGaussianBasis,1)];    %generated augumented vector

        %least square solution with non-negativity constraint
        [GaussianCoefficients,resnorm,residual,exitflag,output] = ...
        lsqnonneg(AugWeightMatrix,AugMeasuredSignals);

        HiResReconstructedSpectrum(:,i)=HiResGaussianBasis*GaussianCoefficients;                                %reconstructed spectrum for plotting
        HiResReconstructedSpectrum(:,i)=HiResReconstructedSpectrum(:,i)/max(HiResReconstructedSpectrum(:,i));   %normalization
        SimulatedSignals(:,i)=HiResResponseCurves*HiResGaussianBasis*GaussianCoefficients;                      %simulated signal with reconstructed spectrum
        
        MeasuResidual(k,i)=norm(SimulatedSignals(:,i)-MeasuredSignals(:,i))^2;        %2-norm of the signal residual
        ReguTerm(k,i)=norm(LaplacianMatrix*GaussianCoefficients)^2;                   %2-norm of the regularization term

    end

    [MinimumResidual,MinimumResidualIndex] = min(MeasuResidual);        %FWHM for the minimum signal residual
    OptimalFWHM(i)=FWHMtemp(MinimumResidualIndex(i));                   %Optimal FWHM that minimize the signal residual

    OptimalGaussianSigma=OptimalFWHM(i)/(2*sqrt(2*log(2)));             %standard deviation of the Gaussian basis, nomalized

            for j=1:NumGaussianBasis                                    %generate all the Guassian basis
                HiResGaussianBasis(:,j)=exp(-0.5*((VecOfLambdasPlot-GaussianCenter(j))/OptimalGaussianSigma).^2);
                HiResGaussianBasis(:,j)=1/OptimalGaussianSigma/sqrt(2*pi)*HiResGaussianBasis(:,j);
            end

        %**************reconstruction with matlab function 'lsqnonneg'
        LaplacianMatrix=eye(NumGaussianBasis);                              %generate Laplacian matrix, which is an identity matrix
        OptimalWeightMatrix=HiResResponseCurves*HiResGaussianBasis;         %this code is equivalent to Eq. (S6) in Science paper

        OptimalGamma=FindOptGamma_GCV(MeasuredSignals(:,i),OptimalWeightMatrix);   %find optimal gamma here with GCV method

        OptimalAugWeightMatrix=[OptimalWeightMatrix;OptimalGamma^2*LaplacianMatrix];      %generate augumented matrix
        OptimalAugMeasuredSignals=[MeasuredSignals(:,i);zeros(NumGaussianBasis,1)];       %generated augumented vector

        HiResReconstructedSpectrum(:,i)=HiResGaussianBasis*GaussianCoefficients;                                %reconstructed spectrum for plotting
        HiResReconstructedSpectrum(:,i)=HiResReconstructedSpectrum(:,i)/max(HiResReconstructedSpectrum(:,i));   %normalization
        SimulatedSignals(:,i)=HiResResponseCurves*HiResGaussianBasis*GaussianCoefficients;                      %simulated signal with reconstructed spectrum

end

    %data visualization
    figure('color','white','position',[0 0 1200 900]);
    subplot(2,2,1);
    semilogy(FWHM,MeasuResidual,'*');hold on;
    xlabel('FWHM');ylabel('||S^{m}-S^{s}||^2'); %plot 2-norm of signal residual vs. FWHM
    title('2-norm of signal residual');
    subplot(2,2,2);
    semilogy(FWHM,ReguTerm,'*');hold on;
    xlabel('FWHM');ylabel('||\alpha||^2'); %plot 2-norm of regularization term vs. FWHM
    title('2-norm of regularization term');
    subplot(2,2,3);
    %plot(MeasuredSignals,'-r');hold on;plot(SimulatedSignals,'*');hold on;
    %legend('Measured Signals','Simulated Signals');
    %legend('Simulated Signals');
    %xlabel('V_g index');ylabel('a.u.');
    %title('Comparison between measured and simulated signals');
    plot(SimulatedSignals,'*');hold on;
    xlabel('V_g index');ylabel('a.u.');
    title('Simulated signals');
    subplot(2,2,4);
    plot(VecOfLambdasPlot*MaxLambda,HiResReconstructedSpectrum,'*');hold on;
    title('Reconstructed spectrum');
    xlabel('\lambda (nm)');ylabel('a.u.');
    %title(['Spectrum, FWHM=' num2str(FWHM(k))]);ylabel('a.u.');xlabel('\lambda (nm)')
    %plot(CSRefSpectrum(:,1),CSRefSpectrum(:,2),'-b');legend('SJS','CS')
   
    %**************GCV method, see: Discrete inverse problems insight and algorithms, pp96.
    function lambda = FindOptGamma_GCV(c,A)
        [U,S,~] = svd(A);
        sigma = diag(S);
        options = optimset('Display','iter');
        InitialGuess=1e-6;  %initial value of Gamma
        lambda = fmincon(@(x) computeGCV(x,U,sigma,c,size(A,1)),...
           InitialGuess,[],[],[],[],0,[],[],options);
        assert(isscalar(lambda)); % check output
        function G = computeGCV(Gamma,U,sigma,b,m)
            numOfcol = length(sigma);
            fi = sigma.^2./(sigma.^2+Gamma^2); % compute filter factor
            rho = 0;
            for i = 1:numOfcol 
                beta = U(:,i)'*b;
                % compute the residual norm
                rho = rho + ((1-fi(i))*beta).^2;
            end
            G = rho/(m-sum(fi))^2;
        end
    end
