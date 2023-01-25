outputFolder = fullfile('tympanic_membrane_orig');
rootFolder = fullfile(outputFolder, 'dataset');
fileFolder = fullfile(rootFolder, 'normal_img');

outputFolder2 = fullfile('dataset_augmented');
rootFolder2 = fullfile(outputFolder2, 'dataset');
fileFolder2 = fullfile(rootFolder2, 'normal_img');

S = dir(fullfile(rootFolder,'normal_img', '*.png'));
length(S)
fprintf('Found %d PNG files.\n', length(S));
hFig = figure;
hFig.WindowState = 'maximized';

for k = 1 : numel(S)
	originalFilename = fullfile(fileFolder, S(k).name);
	[~, baseFileNameNoExt, ~] = fileparts(lower(originalFilename));
	originalImage = imread(originalFilename);
	fprintf('\nRead in %s.\n', originalFilename);
	
    filename = fullfile(fileFolder, S(k).name);
    rgbImg     = imread(filename);
   
    J1 = jitterColorHSV(rgbImg,'Contrast',0.1,'Hue',0.01,'Saturation',0.1,'Brightness',0.1);
    [~,name,~] = fileparts(filename);
    gsFilename = sprintf('%s_cj1.png', name);
    fullFileName = fullfile(fileFolder2, gsFilename);
    imwrite(J1, fullFileName);
    fprintf('    Wrote out %s.\n', fullFileName)
    
    J2 = jitterColorHSV(rgbImg,'Contrast',0.2,'Hue',0.01,'Saturation',0.2,'Brightness',0.1);
    [~,name,~] = fileparts(filename);
    gsFilename = sprintf('%s_cj2.png', name);
    fullFileName = fullfile(fileFolder2, gsFilename);
    imwrite(J2, fullFileName);
    fprintf('    Wrote out %s.\n', fullFileName)
    
    J3 = jitterColorHSV(rgbImg,'Contrast',0.3,'Hue',0.01,'Saturation',0.3,'Brightness',0.1);
    [~,name,~] = fileparts(filename);
    gsFilename = sprintf('%s_cj3.png', name);
    fullFileName = fullfile(fileFolder2, gsFilename);
    imwrite(J3, fullFileName);
    fprintf('    Wrote out %s.\n', fullFileName);

	
end
fprintf('Finished Augmenting');
close(hFig);