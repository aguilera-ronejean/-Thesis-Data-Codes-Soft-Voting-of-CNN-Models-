outputFolder = fullfile('dataset_augmented_undersampled/');
rootFolder = fullfile(outputFolder, 'dataset');
fileFolder = fullfile(rootFolder, 'earwax');

outputFolder2 = fullfile('dataset_augmented_undersampled_histeq');
rootFolder2 = fullfile(outputFolder2, 'dataset');
fileFolder2 = fullfile(rootFolder2, 'earwax');

S = dir(fullfile(rootFolder,'earwax', '*.png'));
fprintf('Found %d PNG files.\n', length(S));
hFig = figure;
hFig.WindowState = 'maximized';

for k = 1 : numel(S)
	originalFilename = fullfile(fileFolder, S(k).name);
	[~, baseFileNameNoExt, ~] = fileparts(lower(originalFilename));
	originalImage = imread(originalFilename);
	fprintf('\nRead in %s.\n', originalFilename);
	
    filename = fullfile(fileFolder, S(k).name);
    
    rgbImg = imread(filename);
    hsv1 = rgb2hsv(rgbImg);
    v1 = hsv1(:,:,3);
    v1 = histeq(v1);
    hsv1(:,:,3) = v1;
    HSimg = hsv2rgb(hsv1);

    [~,name,~] = fileparts(filename);
    gsFilename = sprintf('%s_HISTEQ.png', name);
    fullFileName = fullfile(fileFolder2, gsFilename);
    imwrite(HSimg, fullFileName);
    fprintf('    Wrote out %s.\n', fullFileName);

   % fprintf('Now processing %s...\n', imds.S{k});
    %thisFileName = imds.S{k};  % Note: this includes the folder prepended.
    

    
	
end
close(hFig);