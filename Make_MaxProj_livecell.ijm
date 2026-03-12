// Make_MaxProj_livecell
//
// Make maximum projection of all files opened from a .lif file
//
// adapted from the version of Gert van Cappellen
// 22-07-2024

nframes = 288; // minimum number of frames of the imaging used to filter out other images from the .lif file

macroName="Make_MaxProj_livecell";
if (isOpen("Log")) { 
     selectWindow("Log"); 
     run("Close"); 
} 
print("Macro: "+macroName);
print("---------------------------------------------------------");
dir=getDirectory("Select the input (containing .lif file) directory");
print("Input directory: "+dir);
fileNames=getFileList(dir);
Array.sort(fileNames);

location = dir+"maxProj\\";
File.makeDirectory(location);
getDateAndTime(year, month, dayOfWeek, dayOfMonth, hour, minute, second, msec);
print("Start  Date: ",dayOfMonth,"-",month+1,"-",year,"   Time: ",hour,":",minute,":",second);

for (ii=0; ii<fileNames.length; ii++){
	if(endsWith(fileNames[ii],".lif")){
		name = substring(fileNames[ii], 0, lengthOf(fileNames[ii]) - 4);  //remove the ".lif" in the filename
		File.makeDirectory(location+name);
		run("Bio-Formats Importer", "open=[" + dir+fileNames[ii] + "] autoscale color_mode=Default open_all_series");
		
		// store the image titles. This is needed as "selectImage" can't handle new images opened by "split channels"
		imageTitles = newArray(nImages);
		for (mm=1; mm<=nImages; mm++){
			selectImage(mm);
			imageTitles[mm-1] = getTitle();
		}
		
		// iterate over all the titles to save them after splitting channels and max projection
		for (nn=0; nn<imageTitles.length; nn++){	
			selectWindow(imageTitles[nn]);	
			getDimensions(width, height, channels, slices, frames);
			print(width, height, channels, slices, frames);
			if (frames >= nframes){	
				print("Processing: ",imageTitles[nn]);

				parts = split(imageTitles[nn], "/");
				pos = parts[lengthOf(parts) - 1];  // get the position from the filename
				//imt = replace(imageTitles[nn],"/","_");
				//imt = replace(imt,"\\","_");
				imt = name + "\\" + pos;
				File.makeDirectory(location+imt);
					
				run("Split Channels");
				selectWindow("C1-"+imageTitles[nn]);
				run("Z Project...", "projection=[Max Intensity] all");
				run("Image Sequence... ", "select=["+location+imt+"] dir=["+location+imt+"] format=TIFF name=[]");
			}
		}
		run("Close All");
	}
}
run("Close All");
print("---------------------");
print("Macro correctly ended");
