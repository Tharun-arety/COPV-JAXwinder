# COPV Production Readiness Report

## Status
- Program status: `planning_scaffold_only`
- Source label: `outputs/winding_first_layout.json`
- Layout type: `winding_process`

## Geometry
- Outer radius: `100.0` mm
- Cylinder length: `220.0` mm
- Wall thickness: `8.0` mm
- Opening radius: `10.0` mm
- Pressure case: `6.85`

## Screening Snapshot
- Case: `winding_first`
- FI max with margin: `0.4971662163734436`
- Peak required friction coefficient: `0.14848113059997559`
- Allowable friction coefficient: `0.15`
- Mass delta vs baseline: `756.9090574408309` %
- Burst factor: `1.4180350313595813`

## Exported Program Basis
- Helical course pairs exported: `176`
- Individual helical courses exported: `352`
- Hoop rings exported: `24`
- Discrete cut/restart events: `0`
- Discrete helical pass RMSE: `5.617333549722722e-16`
- Discrete helical activation imbalance CV: `0.0`
- First helical course pair id: `HEL_PAIR_L01_P01_S01`
- First helical course pair length: `471.3898135030256` mm

## Blocking Requirements Before Real Production Release
- [ ] Populate the towpreg storage temperature limit.
- [ ] Populate the towpreg maximum out-time limit.
- [ ] Populate the validated deposition temperature window for the towpreg.
- [ ] Populate the validated tow tension window for the selected material system.
- [ ] Populate the nominal deposition speed for the target winding machine.
- [ ] Populate the machine maximum head speed.
- [ ] Populate the machine maximum mandrel RPM.
- [ ] Populate the minimum steerable turning radius for the line.
- [ ] Populate the nominal heater setpoint for deposition.
- [ ] Populate the nominal compaction force for the head.
- [ ] Populate the cure cycle definition, including ramp and hold steps.
- [ ] Populate the autofrettage target pressure and hold definition.
- [ ] Populate the liner yield pressure or equivalent autofrettage qualification limit.
- [ ] Populate the allowable gap threshold from the quality plan.
- [ ] Populate the allowable overlap threshold from the quality plan.
- [ ] Populate the allowable wrinkle threshold from the quality plan.
- [ ] Populate the final NDI method required for release.
- [ ] Populate the coupon qualification dataset path or evidence reference.
- [ ] Populate the subcomponent qualification dataset path or evidence reference.
- [ ] Populate the vessel-level qualification dataset path or evidence reference.
- [ ] Replace continuous pass-density optimization variables with discrete course scheduling variables, and score cuts/restarts directly inside the optimizer rather than only in downstream planning.
- [ ] Add machine-axis inverse kinematics and NC/post-processor output for the target line.
- [ ] Add an as-built thickness/defect model so structural analysis runs on predicted manufactured state.
- [ ] Couple cure, residual stress, and autofrettage into the structural workflow before release.
- [ ] Correlate the model against coupon, subcomponent, and vessel qualification data.

## Discrete Planning Warnings
- None.

## Planned Execution Sequence
1. Run pair `HEL_PAIR_L01_P01_S01` using `HEL_CW_L01_P01_S01` and `HEL_CCW_L01_P01_S01`.
2. Run pair `HEL_PAIR_L01_P02_S01` using `HEL_CW_L01_P02_S01` and `HEL_CCW_L01_P02_S01`.
3. Run pair `HEL_PAIR_L01_P03_S01` using `HEL_CW_L01_P03_S01` and `HEL_CCW_L01_P03_S01`.
4. Run pair `HEL_PAIR_L01_P04_S01` using `HEL_CW_L01_P04_S01` and `HEL_CCW_L01_P04_S01`.
5. Run pair `HEL_PAIR_L02_P01_S01` using `HEL_CW_L02_P01_S01` and `HEL_CCW_L02_P01_S01`.
6. Run pair `HEL_PAIR_L02_P02_S01` using `HEL_CW_L02_P02_S01` and `HEL_CCW_L02_P02_S01`.
7. Run pair `HEL_PAIR_L02_P03_S01` using `HEL_CW_L02_P03_S01` and `HEL_CCW_L02_P03_S01`.
8. Run pair `HEL_PAIR_L02_P04_S01` using `HEL_CW_L02_P04_S01` and `HEL_CCW_L02_P04_S01`.
9. Run pair `HEL_PAIR_L03_P01_S01` using `HEL_CW_L03_P01_S01` and `HEL_CCW_L03_P01_S01`.
10. Run pair `HEL_PAIR_L03_P02_S01` using `HEL_CW_L03_P02_S01` and `HEL_CCW_L03_P02_S01`.
11. Run pair `HEL_PAIR_L03_P03_S01` using `HEL_CW_L03_P03_S01` and `HEL_CCW_L03_P03_S01`.
12. Run pair `HEL_PAIR_L03_P04_S01` using `HEL_CW_L03_P04_S01` and `HEL_CCW_L03_P04_S01`.
13. Run pair `HEL_PAIR_L04_P01_S01` using `HEL_CW_L04_P01_S01` and `HEL_CCW_L04_P01_S01`.
14. Run pair `HEL_PAIR_L04_P02_S01` using `HEL_CW_L04_P02_S01` and `HEL_CCW_L04_P02_S01`.
15. Run pair `HEL_PAIR_L04_P03_S01` using `HEL_CW_L04_P03_S01` and `HEL_CCW_L04_P03_S01`.
16. Run pair `HEL_PAIR_L04_P04_S01` using `HEL_CW_L04_P04_S01` and `HEL_CCW_L04_P04_S01`.
17. Run pair `HEL_PAIR_L05_P01_S01` using `HEL_CW_L05_P01_S01` and `HEL_CCW_L05_P01_S01`.
18. Run pair `HEL_PAIR_L05_P02_S01` using `HEL_CW_L05_P02_S01` and `HEL_CCW_L05_P02_S01`.
19. Run pair `HEL_PAIR_L05_P03_S01` using `HEL_CW_L05_P03_S01` and `HEL_CCW_L05_P03_S01`.
20. Run pair `HEL_PAIR_L05_P04_S01` using `HEL_CW_L05_P04_S01` and `HEL_CCW_L05_P04_S01`.
21. Run pair `HEL_PAIR_L06_P01_S01` using `HEL_CW_L06_P01_S01` and `HEL_CCW_L06_P01_S01`.
22. Run pair `HEL_PAIR_L06_P02_S01` using `HEL_CW_L06_P02_S01` and `HEL_CCW_L06_P02_S01`.
23. Run pair `HEL_PAIR_L06_P03_S01` using `HEL_CW_L06_P03_S01` and `HEL_CCW_L06_P03_S01`.
24. Run pair `HEL_PAIR_L06_P04_S01` using `HEL_CW_L06_P04_S01` and `HEL_CCW_L06_P04_S01`.
25. Run pair `HEL_PAIR_L07_P01_S01` using `HEL_CW_L07_P01_S01` and `HEL_CCW_L07_P01_S01`.
26. Run pair `HEL_PAIR_L07_P02_S01` using `HEL_CW_L07_P02_S01` and `HEL_CCW_L07_P02_S01`.
27. Run pair `HEL_PAIR_L07_P03_S01` using `HEL_CW_L07_P03_S01` and `HEL_CCW_L07_P03_S01`.
28. Run pair `HEL_PAIR_L07_P04_S01` using `HEL_CW_L07_P04_S01` and `HEL_CCW_L07_P04_S01`.
29. Run pair `HEL_PAIR_L08_P01_S01` using `HEL_CW_L08_P01_S01` and `HEL_CCW_L08_P01_S01`.
30. Run pair `HEL_PAIR_L08_P02_S01` using `HEL_CW_L08_P02_S01` and `HEL_CCW_L08_P02_S01`.
31. Run pair `HEL_PAIR_L08_P03_S01` using `HEL_CW_L08_P03_S01` and `HEL_CCW_L08_P03_S01`.
32. Run pair `HEL_PAIR_L08_P04_S01` using `HEL_CW_L08_P04_S01` and `HEL_CCW_L08_P04_S01`.
33. Run pair `HEL_PAIR_L09_P01_S01` using `HEL_CW_L09_P01_S01` and `HEL_CCW_L09_P01_S01`.
34. Run pair `HEL_PAIR_L09_P02_S01` using `HEL_CW_L09_P02_S01` and `HEL_CCW_L09_P02_S01`.
35. Run pair `HEL_PAIR_L09_P03_S01` using `HEL_CW_L09_P03_S01` and `HEL_CCW_L09_P03_S01`.
36. Run pair `HEL_PAIR_L09_P04_S01` using `HEL_CW_L09_P04_S01` and `HEL_CCW_L09_P04_S01`.
37. Run pair `HEL_PAIR_L10_P01_S01` using `HEL_CW_L10_P01_S01` and `HEL_CCW_L10_P01_S01`.
38. Run pair `HEL_PAIR_L10_P02_S01` using `HEL_CW_L10_P02_S01` and `HEL_CCW_L10_P02_S01`.
39. Run pair `HEL_PAIR_L10_P03_S01` using `HEL_CW_L10_P03_S01` and `HEL_CCW_L10_P03_S01`.
40. Run pair `HEL_PAIR_L10_P04_S01` using `HEL_CW_L10_P04_S01` and `HEL_CCW_L10_P04_S01`.
41. Run pair `HEL_PAIR_L11_P01_S01` using `HEL_CW_L11_P01_S01` and `HEL_CCW_L11_P01_S01`.
42. Run pair `HEL_PAIR_L11_P02_S01` using `HEL_CW_L11_P02_S01` and `HEL_CCW_L11_P02_S01`.
43. Run pair `HEL_PAIR_L11_P03_S01` using `HEL_CW_L11_P03_S01` and `HEL_CCW_L11_P03_S01`.
44. Run pair `HEL_PAIR_L11_P04_S01` using `HEL_CW_L11_P04_S01` and `HEL_CCW_L11_P04_S01`.
45. Run pair `HEL_PAIR_L12_P01_S01` using `HEL_CW_L12_P01_S01` and `HEL_CCW_L12_P01_S01`.
46. Run pair `HEL_PAIR_L12_P02_S01` using `HEL_CW_L12_P02_S01` and `HEL_CCW_L12_P02_S01`.
47. Run pair `HEL_PAIR_L12_P03_S01` using `HEL_CW_L12_P03_S01` and `HEL_CCW_L12_P03_S01`.
48. Run pair `HEL_PAIR_L12_P04_S01` using `HEL_CW_L12_P04_S01` and `HEL_CCW_L12_P04_S01`.
49. Run pair `HEL_PAIR_L13_P01_S01` using `HEL_CW_L13_P01_S01` and `HEL_CCW_L13_P01_S01`.
50. Run pair `HEL_PAIR_L13_P02_S01` using `HEL_CW_L13_P02_S01` and `HEL_CCW_L13_P02_S01`.
51. Run pair `HEL_PAIR_L13_P03_S01` using `HEL_CW_L13_P03_S01` and `HEL_CCW_L13_P03_S01`.
52. Run pair `HEL_PAIR_L13_P04_S01` using `HEL_CW_L13_P04_S01` and `HEL_CCW_L13_P04_S01`.
53. Run pair `HEL_PAIR_L14_P01_S01` using `HEL_CW_L14_P01_S01` and `HEL_CCW_L14_P01_S01`.
54. Run pair `HEL_PAIR_L14_P02_S01` using `HEL_CW_L14_P02_S01` and `HEL_CCW_L14_P02_S01`.
55. Run pair `HEL_PAIR_L14_P03_S01` using `HEL_CW_L14_P03_S01` and `HEL_CCW_L14_P03_S01`.
56. Run pair `HEL_PAIR_L14_P04_S01` using `HEL_CW_L14_P04_S01` and `HEL_CCW_L14_P04_S01`.
57. Run pair `HEL_PAIR_L15_P01_S01` using `HEL_CW_L15_P01_S01` and `HEL_CCW_L15_P01_S01`.
58. Run pair `HEL_PAIR_L15_P02_S01` using `HEL_CW_L15_P02_S01` and `HEL_CCW_L15_P02_S01`.
59. Run pair `HEL_PAIR_L15_P03_S01` using `HEL_CW_L15_P03_S01` and `HEL_CCW_L15_P03_S01`.
60. Run pair `HEL_PAIR_L15_P04_S01` using `HEL_CW_L15_P04_S01` and `HEL_CCW_L15_P04_S01`.
61. Run pair `HEL_PAIR_L16_P01_S01` using `HEL_CW_L16_P01_S01` and `HEL_CCW_L16_P01_S01`.
62. Run pair `HEL_PAIR_L16_P02_S01` using `HEL_CW_L16_P02_S01` and `HEL_CCW_L16_P02_S01`.
63. Run pair `HEL_PAIR_L16_P03_S01` using `HEL_CW_L16_P03_S01` and `HEL_CCW_L16_P03_S01`.
64. Run pair `HEL_PAIR_L16_P04_S01` using `HEL_CW_L16_P04_S01` and `HEL_CCW_L16_P04_S01`.
65. Run pair `HEL_PAIR_L17_P01_S01` using `HEL_CW_L17_P01_S01` and `HEL_CCW_L17_P01_S01`.
66. Run pair `HEL_PAIR_L17_P02_S01` using `HEL_CW_L17_P02_S01` and `HEL_CCW_L17_P02_S01`.
67. Run pair `HEL_PAIR_L17_P03_S01` using `HEL_CW_L17_P03_S01` and `HEL_CCW_L17_P03_S01`.
68. Run pair `HEL_PAIR_L17_P04_S01` using `HEL_CW_L17_P04_S01` and `HEL_CCW_L17_P04_S01`.
69. Run pair `HEL_PAIR_L18_P01_S01` using `HEL_CW_L18_P01_S01` and `HEL_CCW_L18_P01_S01`.
70. Run pair `HEL_PAIR_L18_P02_S01` using `HEL_CW_L18_P02_S01` and `HEL_CCW_L18_P02_S01`.
71. Run pair `HEL_PAIR_L18_P03_S01` using `HEL_CW_L18_P03_S01` and `HEL_CCW_L18_P03_S01`.
72. Run pair `HEL_PAIR_L18_P04_S01` using `HEL_CW_L18_P04_S01` and `HEL_CCW_L18_P04_S01`.
73. Run pair `HEL_PAIR_L19_P01_S01` using `HEL_CW_L19_P01_S01` and `HEL_CCW_L19_P01_S01`.
74. Run pair `HEL_PAIR_L19_P02_S01` using `HEL_CW_L19_P02_S01` and `HEL_CCW_L19_P02_S01`.
75. Run pair `HEL_PAIR_L19_P03_S01` using `HEL_CW_L19_P03_S01` and `HEL_CCW_L19_P03_S01`.
76. Run pair `HEL_PAIR_L19_P04_S01` using `HEL_CW_L19_P04_S01` and `HEL_CCW_L19_P04_S01`.
77. Run pair `HEL_PAIR_L20_P01_S01` using `HEL_CW_L20_P01_S01` and `HEL_CCW_L20_P01_S01`.
78. Run pair `HEL_PAIR_L20_P02_S01` using `HEL_CW_L20_P02_S01` and `HEL_CCW_L20_P02_S01`.
79. Run pair `HEL_PAIR_L20_P03_S01` using `HEL_CW_L20_P03_S01` and `HEL_CCW_L20_P03_S01`.
80. Run pair `HEL_PAIR_L20_P04_S01` using `HEL_CW_L20_P04_S01` and `HEL_CCW_L20_P04_S01`.
81. Run pair `HEL_PAIR_L21_P01_S01` using `HEL_CW_L21_P01_S01` and `HEL_CCW_L21_P01_S01`.
82. Run pair `HEL_PAIR_L21_P02_S01` using `HEL_CW_L21_P02_S01` and `HEL_CCW_L21_P02_S01`.
83. Run pair `HEL_PAIR_L21_P03_S01` using `HEL_CW_L21_P03_S01` and `HEL_CCW_L21_P03_S01`.
84. Run pair `HEL_PAIR_L21_P04_S01` using `HEL_CW_L21_P04_S01` and `HEL_CCW_L21_P04_S01`.
85. Run pair `HEL_PAIR_L22_P01_S01` using `HEL_CW_L22_P01_S01` and `HEL_CCW_L22_P01_S01`.
86. Run pair `HEL_PAIR_L22_P02_S01` using `HEL_CW_L22_P02_S01` and `HEL_CCW_L22_P02_S01`.
87. Run pair `HEL_PAIR_L22_P03_S01` using `HEL_CW_L22_P03_S01` and `HEL_CCW_L22_P03_S01`.
88. Run pair `HEL_PAIR_L22_P04_S01` using `HEL_CW_L22_P04_S01` and `HEL_CCW_L22_P04_S01`.
89. Run pair `HEL_PAIR_L23_P01_S01` using `HEL_CW_L23_P01_S01` and `HEL_CCW_L23_P01_S01`.
90. Run pair `HEL_PAIR_L23_P02_S01` using `HEL_CW_L23_P02_S01` and `HEL_CCW_L23_P02_S01`.
91. Run pair `HEL_PAIR_L23_P03_S01` using `HEL_CW_L23_P03_S01` and `HEL_CCW_L23_P03_S01`.
92. Run pair `HEL_PAIR_L23_P04_S01` using `HEL_CW_L23_P04_S01` and `HEL_CCW_L23_P04_S01`.
93. Run pair `HEL_PAIR_L24_P01_S01` using `HEL_CW_L24_P01_S01` and `HEL_CCW_L24_P01_S01`.
94. Run pair `HEL_PAIR_L24_P02_S01` using `HEL_CW_L24_P02_S01` and `HEL_CCW_L24_P02_S01`.
95. Run pair `HEL_PAIR_L24_P03_S01` using `HEL_CW_L24_P03_S01` and `HEL_CCW_L24_P03_S01`.
96. Run pair `HEL_PAIR_L24_P04_S01` using `HEL_CW_L24_P04_S01` and `HEL_CCW_L24_P04_S01`.
97. Run pair `HEL_PAIR_L25_P01_S01` using `HEL_CW_L25_P01_S01` and `HEL_CCW_L25_P01_S01`.
98. Run pair `HEL_PAIR_L25_P02_S01` using `HEL_CW_L25_P02_S01` and `HEL_CCW_L25_P02_S01`.
99. Run pair `HEL_PAIR_L25_P03_S01` using `HEL_CW_L25_P03_S01` and `HEL_CCW_L25_P03_S01`.
100. Run pair `HEL_PAIR_L25_P04_S01` using `HEL_CW_L25_P04_S01` and `HEL_CCW_L25_P04_S01`.
101. Run pair `HEL_PAIR_L26_P01_S01` using `HEL_CW_L26_P01_S01` and `HEL_CCW_L26_P01_S01`.
102. Run pair `HEL_PAIR_L26_P02_S01` using `HEL_CW_L26_P02_S01` and `HEL_CCW_L26_P02_S01`.
103. Run pair `HEL_PAIR_L26_P03_S01` using `HEL_CW_L26_P03_S01` and `HEL_CCW_L26_P03_S01`.
104. Run pair `HEL_PAIR_L26_P04_S01` using `HEL_CW_L26_P04_S01` and `HEL_CCW_L26_P04_S01`.
105. Run pair `HEL_PAIR_L27_P01_S01` using `HEL_CW_L27_P01_S01` and `HEL_CCW_L27_P01_S01`.
106. Run pair `HEL_PAIR_L27_P02_S01` using `HEL_CW_L27_P02_S01` and `HEL_CCW_L27_P02_S01`.
107. Run pair `HEL_PAIR_L27_P03_S01` using `HEL_CW_L27_P03_S01` and `HEL_CCW_L27_P03_S01`.
108. Run pair `HEL_PAIR_L27_P04_S01` using `HEL_CW_L27_P04_S01` and `HEL_CCW_L27_P04_S01`.
109. Run pair `HEL_PAIR_L28_P01_S01` using `HEL_CW_L28_P01_S01` and `HEL_CCW_L28_P01_S01`.
110. Run pair `HEL_PAIR_L28_P02_S01` using `HEL_CW_L28_P02_S01` and `HEL_CCW_L28_P02_S01`.
111. Run pair `HEL_PAIR_L28_P03_S01` using `HEL_CW_L28_P03_S01` and `HEL_CCW_L28_P03_S01`.
112. Run pair `HEL_PAIR_L28_P04_S01` using `HEL_CW_L28_P04_S01` and `HEL_CCW_L28_P04_S01`.
113. Run pair `HEL_PAIR_L29_P01_S01` using `HEL_CW_L29_P01_S01` and `HEL_CCW_L29_P01_S01`.
114. Run pair `HEL_PAIR_L29_P02_S01` using `HEL_CW_L29_P02_S01` and `HEL_CCW_L29_P02_S01`.
115. Run pair `HEL_PAIR_L29_P03_S01` using `HEL_CW_L29_P03_S01` and `HEL_CCW_L29_P03_S01`.
116. Run pair `HEL_PAIR_L29_P04_S01` using `HEL_CW_L29_P04_S01` and `HEL_CCW_L29_P04_S01`.
117. Run pair `HEL_PAIR_L30_P01_S01` using `HEL_CW_L30_P01_S01` and `HEL_CCW_L30_P01_S01`.
118. Run pair `HEL_PAIR_L30_P02_S01` using `HEL_CW_L30_P02_S01` and `HEL_CCW_L30_P02_S01`.
119. Run pair `HEL_PAIR_L30_P03_S01` using `HEL_CW_L30_P03_S01` and `HEL_CCW_L30_P03_S01`.
120. Run pair `HEL_PAIR_L30_P04_S01` using `HEL_CW_L30_P04_S01` and `HEL_CCW_L30_P04_S01`.
121. Run pair `HEL_PAIR_L31_P01_S01` using `HEL_CW_L31_P01_S01` and `HEL_CCW_L31_P01_S01`.
122. Run pair `HEL_PAIR_L31_P02_S01` using `HEL_CW_L31_P02_S01` and `HEL_CCW_L31_P02_S01`.
123. Run pair `HEL_PAIR_L31_P03_S01` using `HEL_CW_L31_P03_S01` and `HEL_CCW_L31_P03_S01`.
124. Run pair `HEL_PAIR_L31_P04_S01` using `HEL_CW_L31_P04_S01` and `HEL_CCW_L31_P04_S01`.
125. Run pair `HEL_PAIR_L32_P01_S01` using `HEL_CW_L32_P01_S01` and `HEL_CCW_L32_P01_S01`.
126. Run pair `HEL_PAIR_L32_P02_S01` using `HEL_CW_L32_P02_S01` and `HEL_CCW_L32_P02_S01`.
127. Run pair `HEL_PAIR_L32_P03_S01` using `HEL_CW_L32_P03_S01` and `HEL_CCW_L32_P03_S01`.
128. Run pair `HEL_PAIR_L32_P04_S01` using `HEL_CW_L32_P04_S01` and `HEL_CCW_L32_P04_S01`.
129. Run pair `HEL_PAIR_L33_P01_S01` using `HEL_CW_L33_P01_S01` and `HEL_CCW_L33_P01_S01`.
130. Run pair `HEL_PAIR_L33_P02_S01` using `HEL_CW_L33_P02_S01` and `HEL_CCW_L33_P02_S01`.
131. Run pair `HEL_PAIR_L33_P03_S01` using `HEL_CW_L33_P03_S01` and `HEL_CCW_L33_P03_S01`.
132. Run pair `HEL_PAIR_L33_P04_S01` using `HEL_CW_L33_P04_S01` and `HEL_CCW_L33_P04_S01`.
133. Run pair `HEL_PAIR_L34_P01_S01` using `HEL_CW_L34_P01_S01` and `HEL_CCW_L34_P01_S01`.
134. Run pair `HEL_PAIR_L34_P02_S01` using `HEL_CW_L34_P02_S01` and `HEL_CCW_L34_P02_S01`.
135. Run pair `HEL_PAIR_L34_P03_S01` using `HEL_CW_L34_P03_S01` and `HEL_CCW_L34_P03_S01`.
136. Run pair `HEL_PAIR_L34_P04_S01` using `HEL_CW_L34_P04_S01` and `HEL_CCW_L34_P04_S01`.
137. Run pair `HEL_PAIR_L35_P01_S01` using `HEL_CW_L35_P01_S01` and `HEL_CCW_L35_P01_S01`.
138. Run pair `HEL_PAIR_L35_P02_S01` using `HEL_CW_L35_P02_S01` and `HEL_CCW_L35_P02_S01`.
139. Run pair `HEL_PAIR_L35_P03_S01` using `HEL_CW_L35_P03_S01` and `HEL_CCW_L35_P03_S01`.
140. Run pair `HEL_PAIR_L35_P04_S01` using `HEL_CW_L35_P04_S01` and `HEL_CCW_L35_P04_S01`.
141. Run pair `HEL_PAIR_L36_P01_S01` using `HEL_CW_L36_P01_S01` and `HEL_CCW_L36_P01_S01`.
142. Run pair `HEL_PAIR_L36_P02_S01` using `HEL_CW_L36_P02_S01` and `HEL_CCW_L36_P02_S01`.
143. Run pair `HEL_PAIR_L36_P03_S01` using `HEL_CW_L36_P03_S01` and `HEL_CCW_L36_P03_S01`.
144. Run pair `HEL_PAIR_L36_P04_S01` using `HEL_CW_L36_P04_S01` and `HEL_CCW_L36_P04_S01`.
145. Run pair `HEL_PAIR_L37_P01_S01` using `HEL_CW_L37_P01_S01` and `HEL_CCW_L37_P01_S01`.
146. Run pair `HEL_PAIR_L37_P02_S01` using `HEL_CW_L37_P02_S01` and `HEL_CCW_L37_P02_S01`.
147. Run pair `HEL_PAIR_L37_P03_S01` using `HEL_CW_L37_P03_S01` and `HEL_CCW_L37_P03_S01`.
148. Run pair `HEL_PAIR_L37_P04_S01` using `HEL_CW_L37_P04_S01` and `HEL_CCW_L37_P04_S01`.
149. Run pair `HEL_PAIR_L38_P01_S01` using `HEL_CW_L38_P01_S01` and `HEL_CCW_L38_P01_S01`.
150. Run pair `HEL_PAIR_L38_P02_S01` using `HEL_CW_L38_P02_S01` and `HEL_CCW_L38_P02_S01`.
151. Run pair `HEL_PAIR_L38_P03_S01` using `HEL_CW_L38_P03_S01` and `HEL_CCW_L38_P03_S01`.
152. Run pair `HEL_PAIR_L38_P04_S01` using `HEL_CW_L38_P04_S01` and `HEL_CCW_L38_P04_S01`.
153. Run pair `HEL_PAIR_L39_P01_S01` using `HEL_CW_L39_P01_S01` and `HEL_CCW_L39_P01_S01`.
154. Run pair `HEL_PAIR_L39_P02_S01` using `HEL_CW_L39_P02_S01` and `HEL_CCW_L39_P02_S01`.
155. Run pair `HEL_PAIR_L39_P03_S01` using `HEL_CW_L39_P03_S01` and `HEL_CCW_L39_P03_S01`.
156. Run pair `HEL_PAIR_L39_P04_S01` using `HEL_CW_L39_P04_S01` and `HEL_CCW_L39_P04_S01`.
157. Run pair `HEL_PAIR_L40_P01_S01` using `HEL_CW_L40_P01_S01` and `HEL_CCW_L40_P01_S01`.
158. Run pair `HEL_PAIR_L40_P02_S01` using `HEL_CW_L40_P02_S01` and `HEL_CCW_L40_P02_S01`.
159. Run pair `HEL_PAIR_L40_P03_S01` using `HEL_CW_L40_P03_S01` and `HEL_CCW_L40_P03_S01`.
160. Run pair `HEL_PAIR_L40_P04_S01` using `HEL_CW_L40_P04_S01` and `HEL_CCW_L40_P04_S01`.
161. Run pair `HEL_PAIR_L41_P01_S01` using `HEL_CW_L41_P01_S01` and `HEL_CCW_L41_P01_S01`.
162. Run pair `HEL_PAIR_L41_P02_S01` using `HEL_CW_L41_P02_S01` and `HEL_CCW_L41_P02_S01`.
163. Run pair `HEL_PAIR_L41_P03_S01` using `HEL_CW_L41_P03_S01` and `HEL_CCW_L41_P03_S01`.
164. Run pair `HEL_PAIR_L41_P04_S01` using `HEL_CW_L41_P04_S01` and `HEL_CCW_L41_P04_S01`.
165. Run pair `HEL_PAIR_L42_P01_S01` using `HEL_CW_L42_P01_S01` and `HEL_CCW_L42_P01_S01`.
166. Run pair `HEL_PAIR_L42_P02_S01` using `HEL_CW_L42_P02_S01` and `HEL_CCW_L42_P02_S01`.
167. Run pair `HEL_PAIR_L42_P03_S01` using `HEL_CW_L42_P03_S01` and `HEL_CCW_L42_P03_S01`.
168. Run pair `HEL_PAIR_L42_P04_S01` using `HEL_CW_L42_P04_S01` and `HEL_CCW_L42_P04_S01`.
169. Run pair `HEL_PAIR_L43_P01_S01` using `HEL_CW_L43_P01_S01` and `HEL_CCW_L43_P01_S01`.
170. Run pair `HEL_PAIR_L43_P02_S01` using `HEL_CW_L43_P02_S01` and `HEL_CCW_L43_P02_S01`.
171. Run pair `HEL_PAIR_L43_P03_S01` using `HEL_CW_L43_P03_S01` and `HEL_CCW_L43_P03_S01`.
172. Run pair `HEL_PAIR_L43_P04_S01` using `HEL_CW_L43_P04_S01` and `HEL_CCW_L43_P04_S01`.
173. Run pair `HEL_PAIR_L44_P01_S01` using `HEL_CW_L44_P01_S01` and `HEL_CCW_L44_P01_S01`.
174. Run pair `HEL_PAIR_L44_P02_S01` using `HEL_CW_L44_P02_S01` and `HEL_CCW_L44_P02_S01`.
175. Run pair `HEL_PAIR_L44_P03_S01` using `HEL_CW_L44_P03_S01` and `HEL_CCW_L44_P03_S01`.
176. Run pair `HEL_PAIR_L44_P04_S01` using `HEL_CW_L44_P04_S01` and `HEL_CCW_L44_P04_S01`.
177. Run hoop ring `HOOP_L01_S01`.
178. Run hoop ring `HOOP_L02_S01`.
179. Run hoop ring `HOOP_L03_S01`.
180. Run hoop ring `HOOP_L04_S01`.
181. Run hoop ring `HOOP_L05_S01`.
182. Run hoop ring `HOOP_L06_S01`.
183. Run hoop ring `HOOP_L07_S01`.
184. Run hoop ring `HOOP_L08_S01`.
185. Run hoop ring `HOOP_L09_S01`.
186. Run hoop ring `HOOP_L10_S01`.
187. Run hoop ring `HOOP_L11_S01`.
188. Run hoop ring `HOOP_L12_S01`.
189. Run hoop ring `HOOP_L13_S01`.
190. Run hoop ring `HOOP_L14_S01`.
191. Run hoop ring `HOOP_L15_S01`.
192. Run hoop ring `HOOP_L16_S01`.
193. Run hoop ring `HOOP_L17_S01`.
194. Run hoop ring `HOOP_L18_S01`.
195. Run hoop ring `HOOP_L19_S01`.
196. Run hoop ring `HOOP_L20_S01`.
197. Run hoop ring `HOOP_L21_S01`.
198. Run hoop ring `HOOP_L22_S01`.
199. Run hoop ring `HOOP_L23_S01`.
200. Run hoop ring `HOOP_L24_S01`.
201. Execute cure cycle `UNSPECIFIED_CURE`.
202. Perform autofrettage. Note: Autofrettage model not yet coupled in-repo.
203. Run release inspection using `UNSPECIFIED`.

## Notes
- This report is generated from the current winding-first staging artifact.
- It is a production-planning scaffold, not a qualified machine release package.
