#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 16:13:15 2018

@author: che
ukca_setup
give a name of stash code items
"""

def add_ukca_to_iris():
    from iris.fileformats.um_cf_map import STASH_TO_CF, CFName

    # Section 0: Horizontal winds (assume non-rotated grid)
    STASH_TO_CF["m01s00i002"] = CFName("eastward_wind", None, "m s-1")
    STASH_TO_CF["m01s00i003"] = CFName("northward_wind", None, "m s-1")

    # Section 0: Emissions
    STASH_TO_CF["m01s00i121"] = CFName(None,
                                       'tendency_of_atmosphere_mass_content_of_sulfur_dioxide_due_to_volcanic_emission_expressed_as_sulfur',
                                       'kg m-2 s-1')  # IRIS 1.7.2 has wrong units (kg/kg) for this
    STASH_TO_CF["m01s00i309"] = CFName("tendency_of_atmosphere_mass_content_of_isoprene_due_to_emission", None,
                                       "kg m-2 s-1")
    STASH_TO_CF["m01s00i310"] = CFName(
        "tendency_of_atmosphere_mass_content_of_black_carbon_dry_aerosol_due_to_emission_from_fossil_fuel_burning",
        None, "kg m-2 s-1")
    STASH_TO_CF["m01s00i311"] = CFName(
        "tendency_of_atmosphere_mass_content_of_black_carbon_dry_aerosol_due_to_emission_from_biofuel_burning", None,
        "kg m-2 s-1")
    STASH_TO_CF["m01s00i312"] = CFName(
        "tendency_of_atmosphere_mass_content_of_particulate_organic_matter_dry_aerosol_due_to_emission_from_fossil_fuel_burning",
        None, "kg m-2 s-1")
    STASH_TO_CF["m01s00i313"] = CFName(
        "tendency_of_atmosphere_mass_content_of_particulate_organic_matter_dry_aerosol_due_to_emission_from_biofuel_burning",
        None, "kg m-2 s-1")
    STASH_TO_CF["m01s00i314"] = CFName("tendency_of_atmosphere_mass_content_of_monoterpenes_due_to_emission", None,
                                       "kg m-2 s-1")
    STASH_TO_CF["m01s00i316"] = CFName(
        "tendency_of_atmosphere_mass_content_of_black_carbon_dry_aerosol_due_to_emission_from_wildfires", None,
        "kg m-2 s-1")
    STASH_TO_CF["m01s00i317"] = CFName(
        "tendency_of_atmosphere_mass_content_of_particulate_organic_matter_dry_aerosol_due_to_emission_from_wildfires",
        None, "kg m-2 s-1")

    #Section 0: Dynamics
    STASH_TO_CF["m01s00i016"] = CFName(None,"convective cloud liquid water path", "g m-2")
    STASH_TO_CF["m01s00i253"] = CFName(None,"density*R*R","kg m")
    STASH_TO_CF["m01s00i389"] = CFName(None,"dry rho","kg m-3")

    #Section 0: dust
    STASH_TO_CF['m01s00i431'] = CFName(None,'dust division1 mass mixing ratio','kg kg-1')
    STASH_TO_CF['m01s00i432'] = CFName(None,'dust division2 mass mixing ratio','kg kg-1')
    STASH_TO_CF['m01s00i433'] = CFName(None,'dust division3 mass mixing ratio','kg kg-1')
    STASH_TO_CF['m01s00i434'] = CFName(None,'dust division4 mass mixing ratio','kg kg-1')
    STASH_TO_CF['m01s00i435'] = CFName(None,'dust division5 mass mixing ratio','kg kg-1')
    STASH_TO_CF['m01s00i436'] = CFName(None,'dust division6 mass mixing ratio','kg kg-1')

    # Section 1: shortwave forcing
    STASH_TO_CF["m01s01i517"] = CFName("upwelling_shortwave_flux_in_clean_air",None,"W m-2")
    STASH_TO_CF["m01s01i518"] = CFName("downwelling_shortwave_flux_in_clean_air",None,"W m-2")
    STASH_TO_CF["m01s01i519"] = CFName("upwelling_shortwave_flux_in_clear_clean_air",None,"W m-2")
    STASH_TO_CF["m01s01i520"] = CFName("downwelling_shortwave_flux_in_clear_clean_air",None,"W m-2")
    
    # Section 2: longwave forcing
    STASH_TO_CF["m01s02i517"] = CFName("upwelling_longwave_flux_in_clean_air",None,"W m-2")
    STASH_TO_CF["m01s02i518"] = CFName("downwelling_longwave_flux_in_clean_air",None,"W m-2")
    STASH_TO_CF["m01s02i519"] = CFName("upwelling_longwave_flux_in_clear_clean_air",None,"W m-2")
    STASH_TO_CF["m01s02i520"] = CFName("downwelling_longwave_flux_in_clear_clean_air",None,"W m-2")
    

    
#    # Section 2: extinction and absorption profiles
#    STASH_TO_CF["m01s02i114"] = CFName("optical_thickness_of_atmosphere_layer_due_to_ambient_aerosol", None, "1")
#    STASH_TO_CF["m01s02i115"] = CFName("absorption_optical_thickness_of_atmosphere_layer_due_to_ambient_aerosol", None,
#                                       "1")
    # Section 2: AAOD
    STASH_TO_CF["m01s02i240"] = CFName(
        "atmosphere_absorption_optical_thickness_due_to_soluble_aitken_mode_ambient_aerosol", None, "1")
    STASH_TO_CF["m01s02i241"] = CFName(
        "atmosphere_absorption_optical_thickness_due_to_soluble_accumulation_mode_ambient_aerosol", None, "1")
    STASH_TO_CF["m01s02i242"] = CFName(
        "atmosphere_absorption_optical_thickness_due_to_soluble_coarse_mode_ambient_aerosol", None, "1")
    STASH_TO_CF["m01s02i243"] = CFName(
        "atmosphere_absorption_optical_thickness_due_to_insoluble_aitken_mode_ambient_aerosol", None, "1")
    STASH_TO_CF["m01s02i585"] = CFName("atmosphere_absorption_optical_thickness_due_to_dust_ambient_aerosol", None, "1")
    STASH_TO_CF["m01s02i531"] = CFName("ukca_3d_aerosol_absorption_coefficient", None, "m-1")

    # Section 2: AOD
    STASH_TO_CF["m01s02i285"] = CFName("atmosphere_optical_thickness_due_to_dust_ambient_aerosol", None, "1")
    STASH_TO_CF["m01s02i300"] = CFName("atmosphere_optical_thickness_due_to_soluble_aitken_mode_ambient_aerosol", None,
                                       "1")
    STASH_TO_CF["m01s02i301"] = CFName("atmosphere_optical_thickness_due_to_soluble_accumulation_mode_ambient_aerosol",
                                       None, "1")
    STASH_TO_CF["m01s02i302"] = CFName("atmosphere_optical_thickness_due_to_soluble_coarse_mode_ambient_aerosol", None,
                                       "1")
    STASH_TO_CF["m01s02i303"] = CFName("atmosphere_optical_thickness_due_to_insoluble_aitken_mode_ambient_aerosol",
                                       None, "1")
    # Section 2: liquid water path
    STASH_TO_CF["m01s02i391"] = CFName("large_scale_liquid_water_path", None, "kg m-2")
    STASH_TO_CF["m01s02i392"] = CFName("large_scale_ice_water_path", None, "kg m-2")
    STASH_TO_CF["m01s02i393"] = CFName("convective_liquid_water_path", None, "kg m-2")
    STASH_TO_CF["m01s02i394"] = CFName("convective_ice_water_path", None, "kg m-2")
    STASH_TO_CF["m01s02i395"] = CFName("convective_core_liquid_water_path", None, "kg m-2")
    STASH_TO_CF["m01s02i396"] = CFName("convective_core_ice_water_path", None, "kg m-2")
    
    # Setion 2: COSP
    STASH_TO_CF['m01s02i321']= CFName(None,'Mask for CALIPSO low-level cloud faction','1')
    STASH_TO_CF['m01s02i322']= CFName(None,'Mask for CALIPSO mid-level cloud faction','1')
    STASH_TO_CF['m01s02i323']= CFName(None,'Mask for CALIPSO high-level cloud faction','1')
    STASH_TO_CF['m01s02i324']= CFName(None,'Mask for CALIPSO total cloud faction','1')
    STASH_TO_CF['m01s02i330']= CFName(None,'Mask for ISCCP/MISR/MODIS cloud','1')
    STASH_TO_CF['m01s02i331']= CFName(None,'Weighted cloud albedo','1')
    STASH_TO_CF['m01s02i332']= CFName(None,'Weighted cloud optical depth','1')
    STASH_TO_CF['m01s02i333']= CFName(None,'Weighted cloud top pressure','Pa')
    STASH_TO_CF['m01s02i331']= CFName(None,'Weighted cloud albedo','1')
    STASH_TO_CF['m01s02i334']= CFName(None,'ISCCP total cloud area','1')
    STASH_TO_CF['m01s02i344']= CFName(None,'CALIPS low-level cloud fraction','1')
    STASH_TO_CF['m01s02i345']= CFName(None,'CALIPS mid-level cloud fraction','1')
    STASH_TO_CF['m01s02i346']= CFName(None,'CALIPS high-level cloud fraction','1')
    STASH_TO_CF['m01s02i347']= CFName(None,'CALIPS total cloud fraction','1')
    STASH_TO_CF['m01s02i347']= CFName(None,'CALIPS total cloud fraction','1')
    STASH_TO_CF['m01s02i380']= CFName(None,'Large scale cloud liquid effective radius not weighted','m')
    STASH_TO_CF['m01s02i381']= CFName(None,'Large scale cloud ice effective radius not weighted','m')
    STASH_TO_CF['m01s02i382']= CFName(None,'Large scale cloud rainfall effective radius not weighted','m')
    STASH_TO_CF['m01s02i383']= CFName(None,'Large scale cloud snowfall effective radius not weighted','m')

    # Section 3: Near-surface winds (assume non-rotated grid)
    STASH_TO_CF["m01s03i209"] = CFName("eastward_wind", None, "m s-1")
    STASH_TO_CF["m01s03i210"] = CFName("northward_wind", None, "m s-1")
    STASH_TO_CF["m01s00i225"] = CFName("eastward_wind", None, "m s-1")
    STASH_TO_CF["m01s00i226"] = CFName("northward_wind", None, "m s-1")
    
    # Section 15: Dynamics
    STASH_TO_CF["m01s15i271"] = CFName("air_density", None, "kg m-3")
    # STASH_TO_CF["m01s38i504"] = CFName(None,'cloud droplet number concentration','m-3')
    STASH_TO_CF["m01s34i968"] = CFName(None,'cloud droplet number concentration','m-3')
    STASH_TO_CF['m01s03i363'] = CFName(None,'Parametrized entrainment rate at the boundary layer top','m s-1')
    STASH_TO_CF['m01s03i476'] = CFName(None,'Combined bound layer type','1')
    STASH_TO_CF['m01s16i004'] = CFName(None,'Temperature on theta levels','K')
    STASH_TO_CF['m01s38i481'] = CFName(None,'Charac updraft * CDL FLAG','m s-1')

    # Section 34: UKCA Tracers
    STASH_TO_CF["m01s34i101"] = CFName("number_fraction_of_soluble_nucleation_mode_dry_aerosol_in_air", None, "1")
    STASH_TO_CF["m01s34i102"] = CFName("mass_fraction_of_sulfate_soluble_nucleation_mode_dry_aerosol_in_air", None,
                                       "kg kg-1")
    STASH_TO_CF["m01s34i103"] = CFName("number_fraction_of_soluble_aitken_mode_dry_aerosol_in_air", None, "1")
    STASH_TO_CF["m01s34i104"] = CFName("mass_fraction_of_sulfate_soluble_aitken_mode_dry_aerosol_in_air", None,
                                       "kg kg-1")
    STASH_TO_CF["m01s34i105"] = CFName("mass_fraction_of_black_carbon_soluble_aitken_mode_dry_aerosol_in_air", None,
                                       "kg kg-1")
    STASH_TO_CF["m01s34i106"] = CFName(
        "mass_fraction_of_particulate_organic_matter_soluble_aitken_mode_dry_aerosol_in_air", None, "kg kg-1")
    STASH_TO_CF["m01s34i107"] = CFName("number_fraction_of_soluble_accumulation_mode_dry_aerosol_in_air", None, "1")
    STASH_TO_CF["m01s34i108"] = CFName("mass_fraction_of_sulfate_soluble_accumulation_mode_dry_aerosol_in_air", None,
                                       "kg kg-1")
    STASH_TO_CF["m01s34i109"] = CFName("mass_fraction_of_black_carbon_soluble_accumulation_mode_dry_aerosol_in_air",
                                       None, "kg kg-1")
    STASH_TO_CF["m01s34i110"] = CFName(
        "mass_fraction_of_particulate_organic_matter_soluble_accumulation_mode_dry_aerosol_in_air", None, "kg kg-1")
    STASH_TO_CF["m01s34i111"] = CFName("mass_fraction_of_seasalt_soluble_accumulation_mode_dry_aerosol_in_air", None,
                                       "kg kg-1")
    STASH_TO_CF["m01s34i113"] = CFName("number_fraction_of_soluble_coarse_mode_dry_aerosol_in_air", None, "1")
    STASH_TO_CF["m01s34i114"] = CFName("mass_fraction_of_sulfate_soluble_coarse_mode_dry_aerosol_in_air", None,
                                       "kg kg-1")
    STASH_TO_CF["m01s34i115"] = CFName("mass_fraction_of_black_carbon_soluble_coarse_mode_dry_aerosol_in_air", None,
                                       "kg kg-1")
    STASH_TO_CF["m01s34i116"] = CFName(
        "mass_fraction_of_particulate_organic_matter_soluble_coarse_mode_dry_aerosol_in_air", None, "kg kg-1")
    STASH_TO_CF["m01s34i117"] = CFName("mass_fraction_of_seasalt_soluble_coarse_mode_dry_aerosol_in_air", None,
                                       "kg kg-1")
    STASH_TO_CF["m01s34i119"] = CFName("number_fraction_of_insoluble_aitken_mode_dry_aerosol_in_air", None, "1")
    STASH_TO_CF["m01s34i120"] = CFName("mass_fraction_of_black_carbon_insoluble_aitken_mode_dry_aerosol_in_air", None,
                                       "kg kg-1")
    STASH_TO_CF["m01s34i121"] = CFName(
        "mass_fraction_of_particulate_organic_matter_insoluble_aitken_mode_dry_aerosol_in_air", None, "kg kg-1")
    STASH_TO_CF["m01s34i126"] = CFName(
        "mass_fraction_of_particulate_organic_matter_soluble_nucleation_mode_dry_aerosol_in_air", None, "kg kg-1")
    STASH_TO_CF["m01s34i127"] = CFName(
        "mass_fraction_of_seasalt_soluble_aitken_mode_dry_aerosol_in_air", None, "kg kg-1")

    # Section 17: Interactive CLASSIC emissions
    STASH_TO_CF["m01s17i205"] = CFName(
        "tendency_of_atmosphere_mass_content_of_dimethyl_sulfide_due_to_emission_expressed_as_sulfur", None,
        "kg m-2 s-1")

    # Section 34: UKCA chemistry deposition
    STASH_TO_CF["m01s34i454"] = CFName("tendency_of_moles_of_sulfur_dioxide_due_to_dry_deposition", None, "mol s-1")
    STASH_TO_CF["m01s34i455"] = CFName("tendency_of_moles_of_sulfur_dioxide_due_to_wet_deposition", None, "mol s-1")

    # Section 38: UKCA aerosol primary emissions
    STASH_TO_CF["m01s38i201"] = CFName("tendency_of_moles_of_sulfate_soluble_aitken_mode_dry_aerosol_due_to_emission",
                                       None, "mol s-1")
    STASH_TO_CF["m01s38i202"] = CFName(
        "tendency_of_moles_of_sulfate_soluble_accumulation_mode_dry_aerosol_due_to_emission", None, "mol s-1")
    STASH_TO_CF["m01s38i203"] = CFName("tendency_of_moles_of_sulfate_soluble_coarse_mode_dry_aerosol_due_to_emission",
                                       None, "mol s-1")
    STASH_TO_CF["m01s38i204"] = CFName(
        "tendency_of_moles_of_seasalt_soluble_accumulation_mode_dry_aerosol_due_to_emission", None, "mol s-1")
    STASH_TO_CF["m01s38i205"] = CFName("tendency_of_moles_of_seasalt_soluble_coarse_mode_dry_aerosol_due_to_emission",
                                       None, "mol s-1")
    STASH_TO_CF["m01s38i207"] = CFName(
        "tendency_of_moles_of_black_carbon_insoluble_aitken_mode_dry_aerosol_due_to_emission", None, "mol s-1")
    STASH_TO_CF["m01s38i209"] = CFName(
        "tendency_of_moles_of_particulate_organic_matter_insoluble_aitken_mode_dry_aerosol_due_to_emission", None,
        "mol s-1")

    # Section 38: UKCA aerosol dry deposition
    STASH_TO_CF["m01s38i214"] = CFName(
        "tendency_of_moles_of_sulfate_soluble_nucleation_mode_dry_aerosol_due_to_dry_deposition", None, "mol s-1")
    STASH_TO_CF["m01s38i215"] = CFName(
        "tendency_of_moles_of_sulfate_soluble_aitken_mode_dry_aerosol_due_to_dry_deposition", None, "mol s-1")
    STASH_TO_CF["m01s38i216"] = CFName(
        "tendency_of_moles_of_sulfate_soluble_accumulation_mode_dry_aerosol_due_to_dry_deposition", None, "mol s-1")
    STASH_TO_CF["m01s38i217"] = CFName(
        "tendency_of_moles_of_sulfate_soluble_coarse_mode_dry_aerosol_due_to_dry_deposition", None, "mol s-1")
    STASH_TO_CF["m01s38i218"] = CFName(
        "tendency_of_moles_of_seasalt_soluble_accumulation_mode_dry_aerosol_due_to_dry_deposition", None, "mol s-1")
    STASH_TO_CF["m01s38i219"] = CFName(
        "tendency_of_moles_of_seasalt_soluble_coarse_mode_dry_aerosol_due_to_dry_deposition", None, "mol s-1")
    STASH_TO_CF["m01s38i220"] = CFName(
        "tendency_of_moles_of_black_carbon_soluble_aitken_mode_dry_aerosol_due_to_dry_deposition", None, "mol s-1")
    STASH_TO_CF["m01s38i221"] = CFName(
        "tendency_of_moles_of_black_carbon_soluble_accumulation_mode_dry_aerosol_due_to_dry_deposition", None,
        "mol s-1")
    STASH_TO_CF["m01s38i222"] = CFName(
        "tendency_of_moles_of_black_carbon_soluble_coarse_mode_dry_aerosol_due_to_dry_deposition", None, "mol s-1")
    STASH_TO_CF["m01s38i223"] = CFName(
        "tendency_of_moles_of_black_carbon_insoluble_aitken_mode_dry_aerosol_due_to_dry_deposition", None, "mol s-1")
    STASH_TO_CF["m01s38i224"] = CFName(
        "tendency_of_moles_of_particulate_organic_matter_soluble_nucleation_mode_dry_aerosol_due_to_dry_deposition",
        None, "mol s-1")
    STASH_TO_CF["m01s38i225"] = CFName(
        "tendency_of_moles_of_particulate_organic_matter_soluble_aitken_mode_dry_aerosol_due_to_dry_deposition", None,
        "mol s-1")
    STASH_TO_CF["m01s38i226"] = CFName(
        "tendency_of_moles_of_particulate_organic_matter_soluble_accumulation_mode_dry_aerosol_due_to_dry_deposition",
        None, "mol s-1")
    STASH_TO_CF["m01s38i227"] = CFName(
        "tendency_of_moles_of_particulate_organic_matter_soluble_coarse_mode_dry_aerosol_due_to_dry_deposition", None,
        "mol s-1")
    STASH_TO_CF["m01s38i228"] = CFName(
        "tendency_of_moles_of_particulate_organic_matter_insoluble_aitken_mode_dry_aerosol_due_to_dry_deposition", None,
        "mol s-1")

    # Section 38: UKCA aerosol in-cloud wet deposition
    STASH_TO_CF["m01s38i237"] = CFName(
        "tendency_of_moles_of_sulfate_soluble_nucleation_mode_dry_aerosol_due_to_wet_deposition_in_cloud", None,
        "mol s-1")
    STASH_TO_CF["m01s38i238"] = CFName(
        "tendency_of_moles_of_sulfate_soluble_aitken_mode_dry_aerosol_due_to_wet_deposition_in_cloud", None, "mol s-1")
    STASH_TO_CF["m01s38i239"] = CFName(
        "tendency_of_moles_of_sulfate_soluble_accumulation_mode_dry_aerosol_due_to_wet_deposition_in_cloud", None,
        "mol s-1")
    STASH_TO_CF["m01s38i240"] = CFName(
        "tendency_of_moles_of_sulfate_soluble_coarse_mode_dry_aerosol_due_to_wet_deposition_in_cloud", None, "mol s-1")
    STASH_TO_CF["m01s38i241"] = CFName(
        "tendency_of_moles_of_seasalt_soluble_accumulation_mode_dry_aerosol_due_to_wet_deposition_in_cloud", None,
        "mol s-1")
    STASH_TO_CF["m01s38i242"] = CFName(
        "tendency_of_moles_of_seasalt_soluble_coarse_mode_dry_aerosol_due_to_wet_deposition_in_cloud", None, "mol s-1")
    STASH_TO_CF["m01s38i243"] = CFName(
        "tendency_of_moles_of_black_carbon_soluble_aitken_mode_dry_aerosol_due_to_wet_deposition_in_cloud", None,
        "mol s-1")
    STASH_TO_CF["m01s38i244"] = CFName(
        "tendency_of_moles_of_black_carbon_soluble_accumulation_mode_dry_aerosol_due_to_wet_deposition_in_cloud", None,
        "mol s-1")
    STASH_TO_CF["m01s38i245"] = CFName(
        "tendency_of_moles_of_black_carbon_soluble_coarse_mode_dry_aerosol_due_to_wet_deposition_in_cloud", None,
        "mol s-1")
    STASH_TO_CF["m01s38i246"] = CFName(
        "tendency_of_moles_of_black_carbon_insoluble_aitken_mode_dry_aerosol_due_to_wet_deposition_in_cloud", None,
        "mol s-1")
    STASH_TO_CF["m01s38i247"] = CFName(
        "tendency_of_moles_of_particulate_organic_matter_soluble_nucleation_mode_dry_aerosol_due_to_wet_deposition_in_cloud",
        None, "mol s-1")
    STASH_TO_CF["m01s38i248"] = CFName(
        "tendency_of_moles_of_particulate_organic_matter_soluble_aitken_mode_dry_aerosol_due_to_wet_deposition_in_cloud",
        None, "mol s-1")
    STASH_TO_CF["m01s38i249"] = CFName(
        "tendency_of_moles_of_particulate_organic_matter_soluble_accumulation_mode_dry_aerosol_due_to_wet_deposition_in_cloud",
        None, "mol s-1")
    STASH_TO_CF["m01s38i250"] = CFName(
        "tendency_of_moles_of_particulate_organic_matter_soluble_coarse_mode_dry_aerosol_due_to_wet_deposition_in_cloud",
        None, "mol s-1")
    STASH_TO_CF["m01s38i251"] = CFName(
        "tendency_of_moles_of_particulate_organic_matter_insoluble_aitken_mode_dry_aerosol_due_to_wet_deposition_in_cloud",
        None, "mol s-1")

    # Section 38: UKCA aerosol below-cloud wet deposition
    STASH_TO_CF["m01s38i261"] = CFName(
        "tendency_of_moles_of_sulfate_soluble_nucleation_mode_dry_aerosol_due_to_wet_deposition_below_cloud", None,
        "mol s-1")
    STASH_TO_CF["m01s38i262"] = CFName(
        "tendency_of_moles_of_sulfate_soluble_aitken_mode_dry_aerosol_due_to_wet_deposition_below_cloud", None,
        "mol s-1")
    STASH_TO_CF["m01s38i263"] = CFName(
        "tendency_of_moles_of_sulfate_soluble_accumulation_mode_dry_aerosol_due_to_wet_deposition_below_cloud", None,
        "mol s-1")
    STASH_TO_CF["m01s38i264"] = CFName(
        "tendency_of_moles_of_sulfate_soluble_coarse_mode_dry_aerosol_due_to_wet_deposition_below_cloud", None,
        "mol s-1")
    STASH_TO_CF["m01s38i265"] = CFName(
        "tendency_of_moles_of_seasalt_soluble_accumulation_mode_dry_aerosol_due_to_wet_deposition_below_cloud", None,
        "mol s-1")
    STASH_TO_CF["m01s38i266"] = CFName(
        "tendency_of_moles_of_seasalt_soluble_coarse_mode_dry_aerosol_due_to_wet_deposition_below_cloud", None,
        "mol s-1")
    STASH_TO_CF["m01s38i267"] = CFName(
        "tendency_of_moles_of_black_carbon_soluble_aitken_mode_dry_aerosol_due_to_wet_deposition_below_cloud", None,
        "mol s-1")
    STASH_TO_CF["m01s38i268"] = CFName(
        "tendency_of_moles_of_black_carbon_soluble_accumulation_mode_dry_aerosol_due_to_wet_deposition_below_cloud",
        None, "mol s-1")
    STASH_TO_CF["m01s38i269"] = CFName(
        "tendency_of_moles_of_black_carbon_soluble_coarse_mode_dry_aerosol_due_to_wet_deposition_below_cloud", None,
        "mol s-1")
    STASH_TO_CF["m01s38i270"] = CFName(
        "tendency_of_moles_of_black_carbon_insoluble_aitken_mode_dry_aerosol_due_to_wet_deposition_below_cloud", None,
        "mol s-1")
    STASH_TO_CF["m01s38i271"] = CFName(
        "tendency_of_moles_of_particulate_organic_matter_soluble_nucleation_mode_dry_aerosol_due_to_wet_deposition_below_cloud",
        None, "mol s-1")
    STASH_TO_CF["m01s38i272"] = CFName(
        "tendency_of_moles_of_particulate_organic_matter_soluble_aitken_mode_dry_aerosol_due_to_wet_deposition_below_cloud",
        None, "mol s-1")
    STASH_TO_CF["m01s38i273"] = CFName(
        "tendency_of_moles_of_particulate_organic_matter_soluble_accumulation_mode_dry_aerosol_due_to_wet_deposition_below_cloud",
        None, "mol s-1")
    STASH_TO_CF["m01s38i274"] = CFName(
        "tendency_of_moles_of_particulate_organic_mattesr_soluble_coarse_mode_dry_aerosol_due_to_wet_deposition_below_cloud",
        None, "mol s-1")
    STASH_TO_CF["m01s38i275"] = CFName(
        "tendency_of_moles_of_particulate_organic_matter_insoluble_aitken_mode_dry_aerosol_due_to_wet_deposition_below_cloud",
        None, "mol s-1")
    #section 38: aerosol diameter
    STASH_TO_CF["m01s38i401"] = CFName(None,'Mean dry diameter for particles in nucleation sol mode','m')
    STASH_TO_CF["m01s38i402"] = CFName(None,'Mean dry diameter for particles in aitken sol mode','m')
    STASH_TO_CF["m01s38i403"] = CFName(None,'Mean dry diameter for particles in accumulation sol mode','m')
    STASH_TO_CF["m01s38i404"] = CFName(None,'Mean dry diameter for particles in coarse sol mode','m')
    STASH_TO_CF["m01s38i405"] = CFName(None,'Mean dry diameter for particles in aitken insol mode','m')
    STASH_TO_CF["m01s38i408"] = CFName(None,'Mean wet diameter for particles in nucleation sol mode','m')
    STASH_TO_CF["m01s38i409"] = CFName(None,'Mean wet diameter for particles in aitken sol mode','m')
    STASH_TO_CF["m01s38i410"] = CFName(None,'Mean wet diameter for particles in accumulation sol mode','m')
    STASH_TO_CF["m01s38i411"] = CFName(None,'Mean wet diameter for particles in coarse sol mode','m')



    # Section 38: UKCA CCN concentrations
    STASH_TO_CF["m01s38i437"] = CFName(None, "condensation_nuclei_number_concentration", "cm-3")
    STASH_TO_CF["m01s38i438"] = CFName(None,
                                       "cloud_condensation_nuclei_number_concentration_accumulation_plus_coarse_modes",
                                       "cm-3")
    STASH_TO_CF["m01s38i439"] = CFName(None,
                                       "cloud_condensation_nuclei_number_concentration_accumulation_plus_coarse_plus_aitken_gt_25r_modes",
                                       "cm-3")
    STASH_TO_CF["m01s38i440"] = CFName(None,
                                       "cloud_condensation_nuclei_number_concentration_accumulation_plus_coarse_plus_aitken_gt_35r_modes",
                                       "cm-3")
    STASH_TO_CF["m01s38i441"] = CFName(None, "cloud_droplet_numer_number_concentration driven by CCN", "cm-3")

    STASH_TO_CF["m01s38i484"] = CFName(None, "cloud_condensation_nuclei_number_concentration_at_fixed_supersaturation",
                                       "m-3")

    #section 38: aerosols number concentration
    STASH_TO_CF["m01s38i504"] = CFName(None,'Aerosol number in the nucleation sol mode','m-3')
    STASH_TO_CF["m01s38i505"] = CFName(None,'Aerosol number in the aitken sol mode','m-3')
    STASH_TO_CF["m01s38i506"] = CFName(None,'Aerosol number in the accumulation sol mode','m-3')
    STASH_TO_CF["m01s38i507"] = CFName(None,'Aerosol number in the coarse sol mode','m-3')
    STASH_TO_CF["m01s38i508"] = CFName(None,'Aerosol number in the aitken insol mode','m-3')
    STASH_TO_CF["m01s38i509"] = CFName(None,'Aerosol number in the accumulation insol mode','m-3')
    STASH_TO_CF["m01s38i510"] = CFName(None,'Aerosol number in the coarse insol mode','m-3')

    # This looks 3D but the vertical dimension actually represents different levels of supersaturation.
    # Model saturation levels, lowest model level will be lowest supersaturation:
    saturation_levels = [0.02, 0.04, 0.06, 0.08,
                         0.1, 0.16, 0.2, 0.23,
                         0.3, 0.33, 0.38, 0.4,
                         0.5, 0.6, 0.75, 0.8,
                         0.85, 1.0, 1.2]
    #3d CCN
    STASH_TO_CF["m01s38i601"] = CFName(None, "cloud_condensation_nuclei_number_concentration_at_0.05%","m-3")
    STASH_TO_CF["m01s38i602"] = CFName(None, "cloud_condensation_nuclei_number_concentration_at_0.07%","m-3")
    STASH_TO_CF["m01s38i603"] = CFName(None, "cloud_condensation_nuclei_number_concentration_at_0.08%","m-3")
    STASH_TO_CF["m01s38i604"] = CFName(None, "cloud_condensation_nuclei_number_concentration_at_0.10%","m-3")
    STASH_TO_CF["m01s38i605"] = CFName(None, "cloud_condensation_nuclei_number_concentration_at_0.12%","m-3")
    STASH_TO_CF["m01s38i606"] = CFName(None, "cloud_condensation_nuclei_number_concentration_at_0.16%","m-3")
    STASH_TO_CF["m01s38i607"] = CFName(None, "cloud_condensation_nuclei_number_concentration_at_0.20%","m-3")
    STASH_TO_CF["m01s38i608"] = CFName(None, "cloud_condensation_nuclei_number_concentration_at_0.22%","m-3")
    STASH_TO_CF["m01s38i609"] = CFName(None, "cloud_condensation_nuclei_number_concentration_at_0.25%","m-3")
    STASH_TO_CF["m01s38i610"] = CFName(None, "cloud_condensation_nuclei_number_concentration_at_0.30%","m-3")
    STASH_TO_CF["m01s38i611"] = CFName(None, "cloud_condensation_nuclei_number_concentration_at_0.35%","m-3")
    STASH_TO_CF["m01s38i612"] = CFName(None, "cloud_condensation_nuclei_number_concentration_at_0.44%","m-3")
    STASH_TO_CF["m01s38i613"] = CFName(None, "cloud_condensation_nuclei_number_concentration_at_0.45%","m-3")
    STASH_TO_CF["m01s38i614"] = CFName(None, "cloud_condensation_nuclei_number_concentration_at_0.54%","m-3")
    STASH_TO_CF["m01s38i615"] = CFName(None, "cloud_condensation_nuclei_number_concentration_at_0.55%","m-3")
    STASH_TO_CF["m01s38i616"] = CFName(None, "cloud_condensation_nuclei_number_concentration_at_0.57%","m-3")
    STASH_TO_CF["m01s38i617"] = CFName(None, "cloud_condensation_nuclei_number_concentration_at_0.60%","m-3")
    STASH_TO_CF["m01s38i618"] = CFName(None, "cloud_condensation_nuclei_number_concentration_at_0.75%","m-3")
    STASH_TO_CF["m01s38i619"] = CFName(None, "cloud_condensation_nuclei_number_concentration_at_1.00%","m-3")
    STASH_TO_CF["m01s38i480"] = CFName(None, "Max Supersaturatiion * cloud flag","%")
    # Section 38: UKCA Partial volume conctrations
    STASH_TO_CF["m01s38i446"] = CFName(None, "partial_volume_concentration_of_sulfate_soluble_aitken_mode", "1")
    
    # Section 38: total aerosol
    STASH_TO_CF["m01s38i520"] = CFName(None, "total_aerosol_SO4_load", "kg m-2")
    STASH_TO_CF["m01s38i544"] = CFName(None, "total_aerosol_H2O_load", "kg m-2")
    STASH_TO_CF["m01s38i525"] = CFName(None, "total_aerosol_BC_load", "kg m-2")
    STASH_TO_CF["m01s38i539"] = CFName(None, "total_aerosol_Sea-salt_load", "kg m-2")
    STASH_TO_CF["m01s38i531"] = CFName(None, "total_aerosol_OC_load", "kg m-2")
    
    #Section 38: cloud flag
    STASH_TO_CF["m01s38i478"] = CFName(None, "cloud_flag","1")

    # Section 50: UKCA standard diagnostic. Probably Kg, but I've not double checked.
    # These are mostly used in non-hydrostatic cases. An alternative is to use density of air, either directly or
    # as calculated by pressure and temperature. I'd also need to check if these are dry or wet air...
    STASH_TO_CF["m01s50i061"] = CFName(None, "tropospheric_mass_of_air", "kg")
    STASH_TO_CF["m01s50i063"] = CFName(None, "mass_of_air", "kg")