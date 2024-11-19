workspace "kiwiGL"
	architecture "x64"

	configurations
	{
		"Debug",
		"Release"
	}

outputdir = "%{cfg.buildcfg}-%{cfg.system}-%{cfg.architecture}"

project "kiwi"
	location "kiwi"
	language "C++"

	targetdir ("bin/" .. outputdir .. "/%{prj.name}")
	objdir ("bin-int/" .. outputdir .. "/%{prj.name}")

	files 
	{
		"./kiwi/include/*.hpp",
		"./kiwi/src/*.cpp",
		"./kiwi/*.cpp"
	}

	includedirs 
	{
		"./vendor/SDL2/include"
	}


	libdirs 
	{
		"./vendor/SDL2/lib/x64"
	}
	
	links 
	{
		"SDL2.lib",
		"SDL2main.lib",
		"SDL2test.lib"
	}

	postbuildcommands  
	{
		"{COPYFILE} ../vendor/SDL2/lib/x64/SDL2.dll " .. "../bin/" .. outputdir .. "/kiwi/", -- copy SDL2 dll into the bin folder
		"{MKDIR} ../bin/" .. outputdir .. "/kiwi/assets", 									 -- make the assets folder if it doesnt already exist
		"{COPYDIR} ../assets/ " .. "../bin/" .. outputdir .. "/kiwi/assets/",                -- move all assets into the bin assets folder
	}

	filter "system:windows"
		cppdialect "C++17"
		systemversion "latest"

	filter "configurations:Debug"
		symbols "On"
		kind "ConsoleApp"

	filter "configurations:Release"
		optimize "On"
		kind "WindowedApp"
		entrypoint "mainCRTStartup"