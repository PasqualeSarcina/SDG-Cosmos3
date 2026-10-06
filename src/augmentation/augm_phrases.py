AUGM_PHRASES = {('sunny', 'day'): {'json_fields': {'lighting': {'conditions': 'Bright, direct sunlight.',
                                                 'direction': 'Directional sunlight from [observed '
                                                              'direction].',
                                                 'shadows': 'Distinct, well-defined cast shadows.',
                                                 'illumination_effect': 'Bright highlights and strong '
                                                                        'light-shadow contrast.'},
                                    'intensity': 'Moderate to strong color saturation.',
                                    'background_setting': 'The sky is clear.'},
                    'caption_fragments': {'lighting': {'conditions': 'The scene is illuminated by bright, '
                                                                     'direct sunlight with a strong, clear '
                                                                     'quality.',
                                                       'direction': 'Sunlight reaches the subjects from '
                                                                    '[observed direction], as indicated by '
                                                                    'the illuminated surfaces and cast '
                                                                    'shadows.',
                                                       'shadows': 'Visible cast shadows have well-defined '
                                                                  'edges and appear distinctly darker than '
                                                                  'the surrounding sunlit surfaces.',
                                                       'illumination_effect': 'Direct sunlight produces '
                                                                              'bright highlights and '
                                                                              'pronounced differences '
                                                                              'between illuminated and '
                                                                              'shaded areas.'},
                                          'intensity': 'Colors appear moderately to strongly saturated, with '
                                                       'vivid accents standing out across the scene.',
                                          'background_setting': 'The sky is clear.'}},
 ('cloudy', 'day'): {'json_fields': {'lighting': {'conditions': 'Soft, diffuse daylight under an overcast '
                                                                'sky.',
                                                  'direction': 'Broad ambient light without a discernible '
                                                               'dominant direction.',
                                                  'shadows': 'Soft, faint shadows.',
                                                  'illumination_effect': 'Even illumination with subdued '
                                                                         'highlights and low contrast.'},
                                     'intensity': 'Muted to moderate color saturation.',
                                     'background_setting': 'The sky is overcast.'},
                     'caption_fragments': {'lighting': {'conditions': 'The scene receives soft, diffuse '
                                                                      'daylight beneath a visibly overcast '
                                                                      'sky.',
                                                        'direction': 'Light is distributed broadly across '
                                                                     'the subjects and surroundings, with no '
                                                                     'clearly discernible dominant '
                                                                     'direction.',
                                                        'shadows': 'Visible shadows are faint and '
                                                                   'soft-edged, blending gradually into the '
                                                                   'surrounding surfaces.',
                                                        'illumination_effect': 'Illumination is relatively '
                                                                               'even, with subdued '
                                                                               'highlights and gentle '
                                                                               'transitions between brighter '
                                                                               'and darker areas.'},
                                           'intensity': 'Colors appear muted to moderately saturated, with '
                                                        'relatively few strongly vivid areas.',
                                           'background_setting': 'The sky is overcast.'}},
 ('rainy', 'day'): {'json_fields': {'lighting': {'conditions': 'Diffuse daylight in visibly rainy '
                                                               'conditions.',
                                                 'direction': 'Broad ambient light without a discernible '
                                                              'dominant direction.',
                                                 'shadows': 'Faint or indistinct cast shadows.',
                                                 'illumination_effect': 'Subdued illumination with '
                                                                        'reflections on wet surfaces.'},
                                    'intensity': 'Mostly subdued colors with occasional saturated accents.',
                                    'background_setting': 'Rain clouds cover the sky.'},
                    'caption_fragments': {'lighting': {'conditions': 'Soft, diffuse daylight illuminates the '
                                                                     'scene under visibly rainy conditions.',
                                                       'direction': 'Ambient light reaches the scene '
                                                                    'broadly, without a clearly discernible '
                                                                    'dominant direction.',
                                                       'shadows': 'Cast shadows are faint or indistinct and '
                                                                  'show little separation from the '
                                                                  'surrounding surfaces.',
                                                       'illumination_effect': 'Overall illumination is '
                                                                              'subdued, while wet surfaces '
                                                                              'introduce scattered '
                                                                              'highlights and reflections.'},
                                          'intensity': 'Most colors appear subdued, although individual '
                                                       'objects retain more noticeable saturation.',
                                          'background_setting': 'Rain clouds cover the sky.'}},
 ('foggy', 'day'): {'json_fields': {'lighting': {'conditions': 'Soft daylight diffused by visible fog.',
                                                 'direction': 'Scattered light without a discernible '
                                                              'dominant direction.',
                                                 'shadows': 'Faint or obscured shadows.',
                                                 'illumination_effect': 'Low contrast and reduced visibility '
                                                                        'in the distance.'},
                                    'intensity': 'Low color saturation, especially in distant areas.',
                                    'background_setting': 'The sky is obscured by fog.'},
                    'caption_fragments': {'lighting': {'conditions': 'Visible fog diffuses the daylight, '
                                                                     'giving the illumination a soft, '
                                                                     'scattered quality.',
                                                       'direction': 'Light appears broadly scattered through '
                                                                    'the fog, with no clearly discernible '
                                                                    'dominant direction.',
                                                       'shadows': 'Shadows are faint or obscured by the fog, '
                                                                  'making their boundaries difficult to '
                                                                  'distinguish.',
                                                       'illumination_effect': 'Overall contrast is low, and '
                                                                              'distant forms become less '
                                                                              'distinct as they fade into '
                                                                              'the illuminated fog.'},
                                          'intensity': 'Colors appear weakly saturated, with distant areas '
                                                       'becoming increasingly muted.',
                                          'background_setting': 'The sky is obscured by fog.'}},
 ('snow_clear', 'day'): {'json_fields': {'lighting': {'conditions': 'Direct sunlight over snow beneath a '
                                                                    'clear sky.',
                                                      'direction': 'Directional sunlight with additional '
                                                                   'light reflected from the snow.',
                                                      'shadows': 'Defined cast shadows on the snow.',
                                                      'illumination_effect': 'Strong highlights and '
                                                                             'reflected light in shaded '
                                                                             'areas.'},
                                         'intensity': 'Mostly neutral colors with more saturated accents.',
                                         'background_setting': 'The sky is clear.'},
                         'caption_fragments': {'lighting': {'conditions': 'Bright, direct sunlight '
                                                                          'illuminates the snow-covered '
                                                                          'scene beneath a visibly clear '
                                                                          'sky.',
                                                            'direction': 'Sunlight reaches the scene from '
                                                                         '[observed direction], while the '
                                                                         'snow reflects additional light '
                                                                         'onto nearby surfaces.',
                                                            'shadows': 'Defined cast shadows extend across '
                                                                       'the snow, separating shaded areas '
                                                                       'from sunlit surfaces.',
                                                            'illumination_effect': 'Snow-covered surfaces '
                                                                                   'produce strong '
                                                                                   'highlights, while '
                                                                                   'reflected light '
                                                                                   'increases the '
                                                                                   'illumination of nearby '
                                                                                   'shaded areas.'},
                                               'intensity': 'The scene contains broad areas of neutral '
                                                            'color, with colored objects providing more '
                                                            'saturated accents.',
                                               'background_setting': 'The sky is clear.'}},
 ('snow_overcast', 'day'): {'json_fields': {'lighting': {'conditions': 'Diffuse daylight over snow beneath '
                                                                       'an overcast sky.',
                                                         'direction': 'Broad ambient light without a '
                                                                      'discernible dominant direction.',
                                                         'shadows': 'Soft, faint shadows on the snow.',
                                                         'illumination_effect': 'Even illumination with '
                                                                                'gentle tonal variations.'},
                                            'intensity': 'Subdued colors with occasional moderately '
                                                         'saturated accents.',
                                            'background_setting': 'The sky is overcast.'},
                            'caption_fragments': {'lighting': {'conditions': 'Soft, diffuse daylight '
                                                                             'illuminates the snow-covered '
                                                                             'surroundings beneath an '
                                                                             'overcast sky.',
                                                               'direction': 'Light is spread broadly across '
                                                                            'the scene, without a clearly '
                                                                            'discernible dominant direction.',
                                                               'shadows': 'Shadows on the snow are faint and '
                                                                          'soft-edged, with few sharply '
                                                                          'defined boundaries.',
                                                               'illumination_effect': 'Illumination is '
                                                                                      'relatively even, with '
                                                                                      'gentle tonal '
                                                                                      'variations across the '
                                                                                      'snow-covered '
                                                                                      'surfaces.'},
                                                  'intensity': 'Colors appear subdued overall, while '
                                                               'individual colored objects introduce '
                                                               'moderate saturation.',
                                                  'background_setting': 'The sky is overcast.'}},
 ('cloudy', 'night'): {'json_fields': {'lighting': {'conditions': 'Artificial lighting beneath a visibly '
                                                                  'cloudy night sky.',
                                                    'direction': 'Localized light from [observed sources or '
                                                                 'directions].',
                                                    'shadows': 'Variable shadows around illuminated areas.',
                                                    'illumination_effect': 'Localized brightness against '
                                                                           'darker surroundings.'},
                                       'intensity': 'Subdued colors with saturated illuminated accents.',
                                       'background_setting': 'The night sky is overcast.'},
                       'caption_fragments': {'lighting': {'conditions': 'Artificial light sources illuminate '
                                                                        'the scene beneath a visibly '
                                                                        'cloud-covered night sky.',
                                                          'direction': 'Light reaches the subjects from '
                                                                       '[observed sources or directions], '
                                                                       'with its direction varying across '
                                                                       'illuminated areas.',
                                                          'shadows': 'Shadows vary in softness and intensity '
                                                                     'around the light sources and become '
                                                                     'difficult to distinguish in darker '
                                                                     'areas.',
                                                          'illumination_effect': 'Localized pools of light '
                                                                                 'create noticeable '
                                                                                 'brightness differences '
                                                                                 'between illuminated '
                                                                                 'surfaces and darker '
                                                                                 'surroundings.'},
                                             'intensity': 'Colors are subdued in darker areas, while '
                                                          'illuminated objects and colored lights show '
                                                          'stronger saturation.',
                                             'background_setting': 'The night sky is overcast.'}},
 ('rainy', 'night'): {'json_fields': {'lighting': {'conditions': 'Artificial lighting in visibly rainy '
                                                                 'nighttime conditions.',
                                                   'direction': 'Localized light with reflections from wet '
                                                                'surfaces.',
                                                   'shadows': 'Indistinct shadows interrupted by surface '
                                                              'reflections.',
                                                   'illumination_effect': 'Bright wet-surface reflections '
                                                                          'against dark surroundings.'},
                                      'intensity': 'Subdued colors with more saturated lights and '
                                                   'reflections.',
                                      'background_setting': 'Rain clouds cover the night sky.'},
                      'caption_fragments': {'lighting': {'conditions': 'Artificial light sources illuminate '
                                                                       'the scene under visibly rainy '
                                                                       'nighttime conditions.',
                                                         'direction': 'Light comes from [observed sources or '
                                                                      'directions] and is also reflected by '
                                                                      'wet surfaces.',
                                                         'shadows': 'Cast shadows are difficult to '
                                                                    'distinguish in darker areas, while '
                                                                    'reflections interrupt their appearance '
                                                                    'on wet surfaces.',
                                                         'illumination_effect': 'Wet surfaces produce bright '
                                                                                'reflections and scattered '
                                                                                'glare that stand out '
                                                                                'against the darker '
                                                                                'surroundings.'},
                                            'intensity': 'Colors appear subdued overall, with stronger '
                                                         'saturation concentrated in colored lights and '
                                                         'their reflections.',
                                            'background_setting': 'Rain clouds cover the night sky.'}},
 ('foggy', 'night'): {'json_fields': {'lighting': {'conditions': 'Artificial lighting diffused by visible '
                                                                 'fog.',
                                                   'direction': 'Light scattered around localized sources.',
                                                   'shadows': 'Softened or obscured shadows.',
                                                   'illumination_effect': 'Diffuse light halos and reduced '
                                                                          'background contrast.'},
                                      'intensity': 'Muted colors with stronger saturation near colored '
                                                   'lights.',
                                      'background_setting': 'The night sky is obscured by fog.'},
                      'caption_fragments': {'lighting': {'conditions': 'Visible fog scatters the artificial '
                                                                       'light, giving the illuminated areas '
                                                                       'a soft, diffuse quality.',
                                                         'direction': 'Light spreads outward from localized '
                                                                      'sources, although the fog makes its '
                                                                      'direction less distinct.',
                                                         'shadows': 'Shadow edges are softened or obscured '
                                                                    'by the fog, with little shadow detail '
                                                                    'discernible in darker areas.',
                                                         'illumination_effect': 'Diffuse halos surround '
                                                                                'visible light sources, '
                                                                                'while background contrast '
                                                                                'and visibility decrease '
                                                                                'through the fog.'},
                                            'intensity': 'Colors appear muted overall, with stronger '
                                                         'saturation concentrated near visible colored light '
                                                         'sources.',
                                            'background_setting': 'The night sky is obscured by fog.'}},
 ('snow_clear', 'night'): {'json_fields': {'lighting': {'conditions': 'Artificial lighting over snow beneath '
                                                                      'a clear night sky.',
                                                        'direction': 'Localized light with additional '
                                                                     'reflections from the snow.',
                                                        'shadows': 'Cast shadows on illuminated snow.',
                                                        'illumination_effect': 'Bright snow-covered areas '
                                                                               'against darker '
                                                                               'surroundings.'},
                                           'intensity': 'Mostly neutral colors with saturated illuminated '
                                                        'accents.',
                                           'background_setting': 'The night sky is clear.'},
                           'caption_fragments': {'lighting': {'conditions': 'Artificial light sources '
                                                                            'illuminate the snow-covered '
                                                                            'scene beneath a visibly clear '
                                                                            'night sky.',
                                                              'direction': 'Light reaches the scene from '
                                                                           '[observed sources or '
                                                                           'directions], while the snow '
                                                                           'reflects additional light onto '
                                                                           'nearby surfaces.',
                                                              'shadows': 'Cast shadows are visible where '
                                                                         'directional light reaches the '
                                                                         'snow, with their definition '
                                                                         'varying around individual sources.',
                                                              'illumination_effect': 'Illuminated snow '
                                                                                     'creates bright areas '
                                                                                     'against darker '
                                                                                     'surroundings and '
                                                                                     'reflects light onto '
                                                                                     'nearby objects.'},
                                                 'intensity': 'Snow-covered areas have little chromatic '
                                                              'saturation, while illuminated colored objects '
                                                              'provide stronger accents.',
                                                 'background_setting': 'The night sky is clear.'}},
 ('snow_overcast', 'night'): {'json_fields': {'lighting': {'conditions': 'Artificial lighting over snow '
                                                                         'beneath an overcast night sky.',
                                                           'direction': 'Localized light reflected by nearby '
                                                                        'snow-covered surfaces.',
                                                           'shadows': 'Variable shadows in illuminated '
                                                                      'areas.',
                                                           'illumination_effect': 'Reflected light across '
                                                                                  'the snow and darker '
                                                                                  'surrounding areas.'},
                                              'intensity': 'Restrained colors with stronger saturation near '
                                                           'colored lights.',
                                              'background_setting': 'The night sky is overcast.'},
                              'caption_fragments': {'lighting': {'conditions': 'Artificial light sources '
                                                                               'illuminate the snow-covered '
                                                                               'scene beneath a visibly '
                                                                               'overcast night sky.',
                                                                 'direction': 'Light comes from [observed '
                                                                              'sources or directions] and is '
                                                                              'reflected across nearby '
                                                                              'snow-covered surfaces.',
                                                                 'shadows': 'Shadows vary around the light '
                                                                            'sources and become less '
                                                                            'discernible in weakly '
                                                                            'illuminated areas.',
                                                                 'illumination_effect': 'Snow reflects the '
                                                                                        'available light, '
                                                                                        'creating '
                                                                                        'illuminated patches '
                                                                                        'separated by darker '
                                                                                        'surrounding areas.'},
                                                    'intensity': 'Colors are generally restrained, with '
                                                                 'stronger saturation concentrated in '
                                                                 'illuminated objects and colored lights.',
                                                    'background_setting': 'The night sky is overcast.'}}}

# Alternatives for indiscernible details and monochrome images.
FALLBACK_PHRASES = {'concise': {'direction': 'No discernible dominant light direction.',
             'shadows': 'No discernible cast shadows.',
             'monochrome_intensity': 'N/A'},
 'discursive': {'direction': 'No dominant lighting direction is clearly discernible across the subjects and '
                             'surroundings.',
                'shadows': 'No distinct cast shadows are discernible on the visible subjects or surrounding '
                           'surfaces.',
                'monochrome_intensity': 'Color saturation is not applicable because the image is '
                                        'monochrome.'}}
