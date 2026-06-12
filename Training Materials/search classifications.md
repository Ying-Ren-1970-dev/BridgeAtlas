Each pdf is treated as one project. The project level information applies to all pages/sheets within the project. Sheet level information applies to the single page/sheet, but also applies to cross referenced sheets and details.



* Project level information

  * General Plan

    * Project information

      * Name of project
      * date of drawing
      * location
      * owner's information
      * designer's information
      * Engineer on record (EOR)
      * Unit
      * % of plans

        * Advanced Planning study (5%)
        * Preliminary Study (15%)
        * Type selection, PED (30%)
        * Detail Engineering Design (65%)
        * 95%
        * Final Engineering Design (100%)
        * Post 100% 
        * As-built
    * structure type: 

      * bridge:

        * &#x20;Superstructure:

          * Concrete

            * CIP Concrete Box Girder
            * CIP PS/PT Box Girder
            * CIP concrete slab
            * precast Concrete Box
            * Precast U tub
            * Precast I girder
            * Precast slab panel
            * Precast beam
            * Cable stayed
            * suspension
            * arch
          * Steel:

            * Truss
            * Tied arch
            * Plate girder
            * W girder
            * Steel I girder
            * Isotropic box
            * Cable stayed
            * suspension
          * Composite
          * Timber
        * Substructure

          * CIP concrete
          * Precast Concrete
          * Steel
          * arch
          * masonry
          * pylon
          * pier wall
        * Foundation

          * Large diameter drilled shaft (5' or larger CIDH)
          * CIDH pile - drilled shaft
          * pile extension
          * pile group with pile cap
          * link beam
          * CISS (Cast in steel shell)
          * Steel pipe
          * steel I 
          * spread footing
          * caisson
          * isolation casing
      * Tunnel

        * Cut and cover
        * bored
        * corrugated metal
      * Retaining wall

        * Caltrans Cast in place concrete

          * Type 1
          * Type 2
          * Type 4
          * Type 5
          * Type 6
          * Type 7
        * U shaped Wall
        * Shoulder pile wall
        * secant wall
        * Soil nail wall
        * Tie back wall
        * Corrugated metal wall
        * stone wall
        * masonry wall
        * cast in place (CIP) concrete
      * Culvert

        * Pipe
        * concrete cox
        * corrugated metal
      * soundwall
      * monument
      * Sign pole
      * building
      * station
    * Structure information for cross referencing:

      * Abutment numbers
      * Bent numbers
      * Pier numbers
      * Span length
      * Span connections (i.e. connect bent 2 to bent 3)
      * Elevation and control line inforamtion
  * General Note sheet

    * Design code

      * AASHTO

        * LRFD BDS
        * Guide spec for pedestrian
        * Steel
        * Seismic
      * CSA Canadian code
      * Caltrans

        * amendment to AASHTO
        * Seismic Design Criteria (SDC)
      * International building code
      * Euro code
    * Seismic

      * PGA
      * zone
    * Live load

      * Pedestrian

        * H10
        * 90psf
      * Vehicular

        * HL-93
        * Permit
        * H20
        * H40
      * Rail

        * Coper 80
        * Light rail
    * Index to plans: can be cross referenced to individual sheet name on title block




* Sheet/Page level information

  * CAD drawing view classification (each chunk on a plan sheet)

    * one chunk = one labeled drawing region (title + scale + drawing body)
    * common types: plan, elevation, layout, detail, section
    * view is uncommon; only when explicitly labeled (VIEW A-A)
    * additional categories are inferred from the label: diagram, as_built, schedule, etc.
    * imprecise labels fall back to section or detail: TYPICAL SECTION, TYPICAL DETAILS, named details without numbers
    * examples: SECTION B-B (section), DETAIL 1 (detail), 24" STEEL PIPE PILE ELEVATION (elevation), BENT CAP CAMBER DIAGRAM (diagram), POST ANCHORAGE DETAIL (detail, general)
    * chunk boundaries are approximate layout guides; leader-line text belongs to the chunk where the leader arrowhead points
    * search equivalence: detail intent also matches section chunks for the same component (e.g. shear key detail also returns shear key section)

  * Cross-referencing between drawing views

    * section, view, and detail can reference plan, elevation, layout, detail, view, or section
    * section cuts are labeled like SECTION A-A, SECTION B-B, SECTION C-C
    * views are labeled like VIEW A-A, VIEW B-B
    * detail callouts are usually dashed circles with leaders and labels like DETAIL 1, DETAIL 2, DETAIL 3
    * when a section, view, or detail is not on the same sheet, sheet notes usually say which sheet/page contains the referenced drawing
    * notes on a sheet supplement the details and sections and often link to other sheets, details, sections, or views

  * Plan sheet groups

    * plan sheets are grouped by structural system: abutment, bent, girder, pier, footing, wingwall, etc.
    * each group may contain one or many sheets
    * example abutment group: Abutment Layout No. 1, Abutment Layout No. 2, Abutment Details No. 1, Abutment Details No. 2, Abutment Details No. 3
    * within a group, section letters and detail numbers usually restart (A-A / Detail 1 at the start of each group)
    * layout sheets may contain plan, elevation, and layout views
    * detail sheets may contain detail, section, and view drawings

  * title block

    * sheet name: can be used for cross reference details and call out and notes. For example, notes on a detail that reference to a sheet, should link the detail to referenced sheet.
    * detail/section: each detail and section is cut from another drawing (elevation, plan, or another section or detail). The origin of the cut would call out the sheet and detail/section number referencing it. Need to link and analyze these information.
    * notes on sheet: notes would supplement information of the details and sections on this sheet. It tells additional information of the detail and section, include but not limited to referring to other sheet and other details. Need to link and analyze these information.



