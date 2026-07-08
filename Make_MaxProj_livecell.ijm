// Make maximum projection of all files opened from a .lif file
//
// it opens the separate positions one by one to prevent memory overflow
// and uses the number of frames to discriminate between timelapses and other files in the .lif file
//

nframes = 140; // minimum number of frames of the imaging used to filter out other images from the .lif file

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
		
		// Count series without actually opening them
		//run("Bio-Formats Importer", "open=[" + dir+fileNames[ii] + "] autoscale color_mode=Default view_metadata only_metadata");
		run("Bio-Formats Macro Extensions");
		Ext.setId(dir + fileNames[ii]);        // initialise the file
		Ext.getSeriesCount(seriesCount);      // get number of series
		print(fileNames[ii] + " -> seriesCount=" + seriesCount);

		for (s = 0; s < seriesCount; s++) {
    		Ext.setSeries(s);                 // choose series (zero-based)
    		Ext.openImagePlus(dir + fileNames[ii]);   // open only that series
	
    		//run("Bio-Formats Importer", "open=[" + dir+fileNames[ii] + "] autoscale color_mode=Default series_" + s);
		
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
					imt = name + File.separator + pos;         
					File.makeDirectory(location + imt);        
					saveDir = location + imt + File.separator + "raw" + File.separator;
					File.makeDirectory(saveDir);
					
					if (channels > 1) {
						run("Split Channels");   // split the channels if the transmission image is also there
						selectWindow("C1-" + imageTitles[nn]);
					}
					else {
						rename("C1-" + imageTitles[nn]);
					}
					
					run("Z Project...", "projection=[Max Intensity] all");					
					selectWindow("MAX_C1-" + imageTitles[nn]);					
					//saveAs("Tiff", saveDir + "stack.tif");
					//run("Image Sequence... ", "select=["+location+imt+"] dir=["+location+imt+"] format=TIFF name=[]");
					run("Image Sequence... ", "select=["+saveDir+"] dir=["+saveDir+"] format=TIFF name=[]");
				}
			}
			
    		run("Close All");
		}
	}
}
run("Close All");
print("---------------------");
print("Macro correctly ended");
