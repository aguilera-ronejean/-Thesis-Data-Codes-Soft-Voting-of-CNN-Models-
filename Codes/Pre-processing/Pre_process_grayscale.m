outputFolder = fullfile('tympanic_membrane_dataset');
rootFolder = fullfile(outputFolder, 'dataset');
fileFolder = fullfile(rootFolder, 'aom');
S = dir(fullfile(rootFolder,'aom', '*.png'));
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
    gsImg      = rgb2gray(rgbImg);

    [~,name,~] = fileparts(filename);
    gsFilename = sprintf('%s_gs.png', name);
    fullFileName = fullfile(fileFolder, gsFilename);
    imwrite(gsImg, fullFileName);
    fprintf('    Wrote out %s.\n', fullFileName);

    
	
end
close(hFig);