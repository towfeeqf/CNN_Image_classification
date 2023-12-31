function result = batchmyimagesresizefcn(inDir, outDir)
%batchmyimagesresizefcn Batch process images using myimagesresizefcn
% RESULT = batchmyimagesresizefcn(INDIR, OUTDIR) processes each file in INDIR
% using the function myimagesresizefcn.
%
% The following fields from the output of myimagesresizefcn are written with their
% corresponding file format to the output directory OUTDIR:
%    bw saved as jpg format
%    imgray saved as jpg format
%
% The following fields are returned in the table RESULT
%
%
% Auto-generated by imageBatchProcessor app on 05-Aug-2022
%----------------------------------------------------------

if(nargin<2)
    outDir = '';
else
    outDir = convertStringsToChars(outDir);
end
if(nargin<1)
    inDir = 'C:\Users\B00804205\Downloads\Image_2\dataimages1\train';
else
    inDir = convertStringsToChars(inDir);
end

includeSubdirectories = true;

% Fields to place in result
workSpaceFields = {

};

% Fields to write out to files. Each entry contains the field name and the
% corresponding file format.
fileFieldsAndFormat = {
     {'im', 'jpg'}
%     {'bw', 'jpg'}
%     {'imgray', 'jpg'}
     };


% All extensions that can be read by IMREAD
imreadFormats       = imformats;
supportedExtensions = [imreadFormats.ext];
% Add dicom extensions
supportedExtensions{end+1} = 'dcm';
supportedExtensions{end+1} = 'ima';
supportedExtensions = strcat('.',supportedExtensions);
% Allow the 'no extension' specification of DICOM
supportedExtensions{end+1} = '';


% Create a image data store that can read all these files
imds = datastore(inDir,...
    'IncludeSubfolders', includeSubdirectories,...
    'Type','image',...
    'FileExtensions',supportedExtensions);
imds.ReadFcn = @readSupportedImage;


% Initialize output (as struct array)
result(numel(imds.Files)) = struct();
% Initialize fields with []
for ind =1:numel(workSpaceFields)
    [result.(workSpaceFields{ind})] = deal([]);
end


% Process each image using myimagesresizefcn
for imgInd = 1:numel(imds.Files)

    inImageFile  = imds.Files{imgInd};

    % Output has the same sub-directory structure as input
    outImageFileWithExtension = strrep(inImageFile, inDir, outDir);
    % Remove the file extension to create the template output file name
    [path, filename,~] = fileparts(outImageFileWithExtension);
    outImageFile = fullfile(path,filename);

    try
        % Read
        im = imds.readimage(imgInd);

        % Process
        oneResult = myimagesresizefcn(im);

        % Accumulate
        for ind = 1:numel(workSpaceFields)
            % Only copy fields specified to be returned in the output
            fieldName = workSpaceFields{ind};
            result(imgInd).(fieldName) = oneResult.(fieldName);
        end

        % Include the input image file name
        result(imgInd).fileName = imds.Files{imgInd};

        % Write chosen fields to image files only if output directory is
        % specified
        if(~isempty(outDir))
            % Create (sub)directory if needed
            outSubDir = fileparts(outImageFile);
            createDirectory(outSubDir);

            for ind = 1:numel(fileFieldsAndFormat)
                fieldName  = fileFieldsAndFormat{ind}{1};
                fileFormat = fileFieldsAndFormat{ind}{2};
                imageData  = oneResult.(fieldName);
                % Add the field name and required file format for this
                % field to the template output file name
                outImageFileWithExtension = [outImageFile,'_',fieldName, '.', fileFormat];

                try
                    imwrite(imageData, outImageFileWithExtension);
                catch IMWRITEFAIL
                    disp(['WRITE FAILED:', inImageFile]);
                    warning(IMWRITEFAIL.identifier, '%s', IMWRITEFAIL.message);
                end
            end
        end

       % disp(['PASSED:', inImageFile]);

    catch READANDPROCESSEXCEPTION
        disp(['FAILED:', inImageFile]);
        warning(READANDPROCESSEXCEPTION.identifier, '%s', READANDPROCESSEXCEPTION.message);
    end

end

result = struct2table(result,'AsArray',true);

end


function img = readSupportedImage(imgFile)
% Image read function with DICOM support
if(isdicom(imgFile))
    img = dicomread(imgFile);
else
    img = imread(imgFile);
end
end

function createDirectory(dirname)
% Make output (sub) directory if needed
if exist(dirname, 'dir')
    return;
end
[success, message] = mkdir(dirname);
if ~success
    disp(['FAILED TO CREATE:', dirname]);
    disp(message);
end
end
