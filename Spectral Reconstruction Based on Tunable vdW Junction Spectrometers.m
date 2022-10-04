%version 2022/10/04, coded by Weiwei Cai, Hoon Hahn Yoon, and Fedor Nigmatulin
clear;clc;close all;

%**************Parameter settings*************
NumWavelengths=101;         %Number of wavelengths for reconstruction
NumWavelengthsInterp=NumWavelengths*10;   %number of interpolated wavelengths for plotting
NumGaussianBasis=101;       %Number of Gaussian Basis
MinLambda=405;              %The smallest wavelength in nm
MaxLambda=845;              %The largest wavelength in nm
FWHMset=10;                	%Full-width half maximum of the Gaussian Basis, normalized with the maximum wavelength and times 1000, unit-less
%FWHMset was manually chosen depending on the incident light spectrum (e.g., whether it is a monochromatic light source or broadband light source).
%FWHMset are 8~13 (in Fig. 2E, monochromatic, 405~845 nm), 85~145 (Fig. 2F, broadband, 405~845 nm), 3~5 (Fig. 3B,E, 675~685 nm), and 150~200 (Fig. 4C, broadband, 405~845 nm).
%Future improvement of automatically selecting FWHM is needed for practical applications, which is beyond the concept demonstration of this paper.

%**************Loading data*************
ResponseFilename=load('CalibrationData.txt');                     %Responsivity as function of wavelength at different Vg (Detectors)
ResponseCurves=ResponseFilename/max(max(ResponseFilename));       %Normalize the input response curve

SignalFilename='MeausuredSignals.txt';
MeasuredSignals=load(SignalFilename);                              %Photocurrent as function of Vg (Detectors)

%**************Initialization of variables*************
NumRow=size(ResponseCurves,1);                                     %Number of Vg's (Detectors)
NumCol=size(MeasuredSignals,2);                                    %Multiple data processing from matrix of MeasuredSignals
SimulatedSignals=zeros(NumRow,NumCol);                             %Initialization of simulated signals
HiResReconstructedSpectrum=zeros(NumWavelengthsInterp,NumCol);     %Initialization of reconstructed spectrum for plotting
    
VecOfLambdas=linspace(MinLambda,MaxLambda,NumWavelengths)'/MaxLambda;             %Normalized wavelengths
VecOfLambdasPlot=linspace(MinLambda,MaxLambda,NumWavelengthsInterp)'/MaxLambda;   %Normalized interpolated wavelengths

HiResResponseCurves=zeros(NumRow,NumWavelengthsInterp);
for r=1:NumRow
    HiResResponseCurves(r,:)=spline(VecOfLambdas,ResponseCurves(r,:),VecOfLambdasPlot);  %Spline interpolation
end
    
MeasuResidual=zeros(1,NumCol);                  %An array that records 2-norm of signal residual
ReguTerm=zeros(1,NumCol);                       %An array that records 2-norm of regularization term

%**************Reconstruction*************
for i=1:NumCol	%Each column constitutes the measured signals for one spectrum to be reconstructed

    MeasuredSignals(:,i)=MeasuredSignals(:,i)/max(max(MeasuredSignals(:,i)));   %Normalizing measured signals

    %**************Generate Gaussian Basis here
    GaussianCenter=linspace(MinLambda,MaxLambda,NumGaussianBasis)'/MaxLambda;   %Normalized wavelength centers
    GaussianBasis=zeros(NumWavelengths,NumGaussianBasis);                       %All the Gaussian basis for reconstruction
    HiResGaussianBasis=zeros(NumWavelengthsInterp,NumGaussianBasis);            %Interpolated Gaussian basis 
        
	GaussianSigma=FWHMset/1000/(2*sqrt(2*log(2)));        %Standard deviation of the Gaussian basis, normalized

	for j=1:NumGaussianBasis                           %Generate all the Guassian basis
		HiResGaussianBasis(:,j)=exp(-0.5*((VecOfLambdasPlot-GaussianCenter(j))/GaussianSigma).^2);
		HiResGaussianBasis(:,j)=1/GaussianSigma/sqrt(2*pi)*HiResGaussianBasis(:,j);
	end

	%**************reconstruction with matlab function 'lsqnonneg'
	LaplacianMatrix=eye(NumGaussianBasis);                  %Generate Laplacian matrix, which is an identity matrix
	WeightMatrix=HiResResponseCurves*HiResGaussianBasis;    %This code is equivalent to Eq. (S6) in the Science paper

	OptimalGamma=FindOptGamma_GCV(MeasuredSignals(:,i),WeightMatrix);       %Find the optimal regularization parameter here with GCV method

	AugWeightMatrix=[WeightMatrix;OptimalGamma^2*LaplacianMatrix];          %Generate augumented matrix
	AugMeasuredSignals=[MeasuredSignals(:,i);zeros(NumGaussianBasis,1)];    %Generated augumented vector

	%least square solution with non-negativity constraint
	[GaussianCoefficients,resnorm,residual,exitflag,output] = ...
	lsqnonneg(AugWeightMatrix,AugMeasuredSignals);

	HiResReconstructedSpectrum(:,i)=HiResGaussianBasis*GaussianCoefficients;                                %Reconstructed spectrum for plotting
	HiResReconstructedSpectrum(:,i)=HiResReconstructedSpectrum(:,i)/max(HiResReconstructedSpectrum(:,i));   %Normalization
	SimulatedSignals(:,i)=HiResResponseCurves*HiResGaussianBasis*GaussianCoefficients;                      %Simulated signal with reconstructed spectrum
	
	MeasuResidual(1,i)=norm(SimulatedSignals(:,i)-MeasuredSignals(:,i))^2;        %2-norm of the signal residual
	ReguTerm(1,i)=norm(LaplacianMatrix*GaussianCoefficients)^2;                   %2-norm of the regularization term

end

%Data visualization
figure('color','white','position',[0 0 1200 900]);
subplot(1,2,1);
plot(SimulatedSignals,'*');hold on;
xlabel('V_g index');ylabel('a.u.');
title('Simulated signals');
subplot(1,2,2);
plot(VecOfLambdasPlot*MaxLambda,HiResReconstructedSpectrum,'*');hold on;
title('Reconstructed spectrum');
xlabel('\lambda (nm)');ylabel('a.u.');

%**************GCV method, see: Discrete inverse problems insight and algorithms, pp. 96.
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
