/* nd_catalog.c: transcription of ports/jibo/runner/src/catalog.rs (see nd_catalog.h), plus the
 * Rust and serde_json behaviour the runner modules share. */
#include "nd_catalog.h"

#include <float.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nd_prompt.h"

/* ─── Unicode tables ───────────────────────────────────────────────────────────────────────────
 * Generated with rustc 1.87.0 by enumerating every char:
 *  nd_dbg_printable: non-ASCII chars that `format!("{:?}", c.to_string())` prints as themselves
 *    (neither grapheme-extended nor non-printable, per core::unicode in 1.87);
 *  nd_white_space: char::is_whitespace.
 * Inclusive ranges, sorted. */
#define ND_DBG_PRINTABLE_N 894u
static const uint32_t nd_dbg_printable[ND_DBG_PRINTABLE_N][2] = {
  {0xA1, 0xAC}, {0xAE, 0x2FF}, {0x370, 0x377}, {0x37A, 0x37F}, {0x384, 0x38A}, {0x38C, 0x38C},
  {0x38E, 0x3A1}, {0x3A3, 0x482}, {0x48A, 0x52F}, {0x531, 0x556}, {0x559, 0x58A}, {0x58D, 0x58F},
  {0x5BE, 0x5BE}, {0x5C0, 0x5C0}, {0x5C3, 0x5C3}, {0x5C6, 0x5C6}, {0x5D0, 0x5EA}, {0x5EF, 0x5F4},
  {0x606, 0x60F}, {0x61B, 0x61B}, {0x61D, 0x64A}, {0x660, 0x66F}, {0x671, 0x6D5}, {0x6DE, 0x6DE},
  {0x6E5, 0x6E6}, {0x6E9, 0x6E9}, {0x6EE, 0x70D}, {0x710, 0x710}, {0x712, 0x72F}, {0x74D, 0x7A5},
  {0x7B1, 0x7B1}, {0x7C0, 0x7EA}, {0x7F4, 0x7FA}, {0x7FE, 0x815}, {0x81A, 0x81A}, {0x824, 0x824},
  {0x828, 0x828}, {0x830, 0x83E}, {0x840, 0x858}, {0x85E, 0x85E}, {0x860, 0x86A}, {0x870, 0x88E},
  {0x8A0, 0x8C9}, {0x903, 0x939}, {0x93B, 0x93B}, {0x93D, 0x940}, {0x949, 0x94C}, {0x94E, 0x950},
  {0x958, 0x961}, {0x964, 0x980}, {0x982, 0x983}, {0x985, 0x98C}, {0x98F, 0x990}, {0x993, 0x9A8},
  {0x9AA, 0x9B0}, {0x9B2, 0x9B2}, {0x9B6, 0x9B9}, {0x9BD, 0x9BD}, {0x9BF, 0x9C0}, {0x9C7, 0x9C8},
  {0x9CB, 0x9CC}, {0x9CE, 0x9CE}, {0x9DC, 0x9DD}, {0x9DF, 0x9E1}, {0x9E6, 0x9FD}, {0xA03, 0xA03},
  {0xA05, 0xA0A}, {0xA0F, 0xA10}, {0xA13, 0xA28}, {0xA2A, 0xA30}, {0xA32, 0xA33}, {0xA35, 0xA36},
  {0xA38, 0xA39}, {0xA3E, 0xA40}, {0xA59, 0xA5C}, {0xA5E, 0xA5E}, {0xA66, 0xA6F}, {0xA72, 0xA74},
  {0xA76, 0xA76}, {0xA83, 0xA83}, {0xA85, 0xA8D}, {0xA8F, 0xA91}, {0xA93, 0xAA8}, {0xAAA, 0xAB0},
  {0xAB2, 0xAB3}, {0xAB5, 0xAB9}, {0xABD, 0xAC0}, {0xAC9, 0xAC9}, {0xACB, 0xACC}, {0xAD0, 0xAD0},
  {0xAE0, 0xAE1}, {0xAE6, 0xAF1}, {0xAF9, 0xAF9}, {0xB02, 0xB03}, {0xB05, 0xB0C}, {0xB0F, 0xB10},
  {0xB13, 0xB28}, {0xB2A, 0xB30}, {0xB32, 0xB33}, {0xB35, 0xB39}, {0xB3D, 0xB3D}, {0xB40, 0xB40},
  {0xB47, 0xB48}, {0xB4B, 0xB4C}, {0xB5C, 0xB5D}, {0xB5F, 0xB61}, {0xB66, 0xB77}, {0xB83, 0xB83},
  {0xB85, 0xB8A}, {0xB8E, 0xB90}, {0xB92, 0xB95}, {0xB99, 0xB9A}, {0xB9C, 0xB9C}, {0xB9E, 0xB9F},
  {0xBA3, 0xBA4}, {0xBA8, 0xBAA}, {0xBAE, 0xBB9}, {0xBBF, 0xBBF}, {0xBC1, 0xBC2}, {0xBC6, 0xBC8},
  {0xBCA, 0xBCC}, {0xBD0, 0xBD0}, {0xBE6, 0xBFA}, {0xC01, 0xC03}, {0xC05, 0xC0C}, {0xC0E, 0xC10},
  {0xC12, 0xC28}, {0xC2A, 0xC39}, {0xC3D, 0xC3D}, {0xC41, 0xC44}, {0xC58, 0xC5A}, {0xC5D, 0xC5D},
  {0xC60, 0xC61}, {0xC66, 0xC6F}, {0xC77, 0xC80}, {0xC82, 0xC8C}, {0xC8E, 0xC90}, {0xC92, 0xCA8},
  {0xCAA, 0xCB3}, {0xCB5, 0xCB9}, {0xCBD, 0xCBE}, {0xCC1, 0xCC1}, {0xCC3, 0xCC4}, {0xCDD, 0xCDE},
  {0xCE0, 0xCE1}, {0xCE6, 0xCEF}, {0xCF1, 0xCF3}, {0xD02, 0xD0C}, {0xD0E, 0xD10}, {0xD12, 0xD3A},
  {0xD3D, 0xD3D}, {0xD3F, 0xD40}, {0xD46, 0xD48}, {0xD4A, 0xD4C}, {0xD4E, 0xD4F}, {0xD54, 0xD56},
  {0xD58, 0xD61}, {0xD66, 0xD7F}, {0xD82, 0xD83}, {0xD85, 0xD96}, {0xD9A, 0xDB1}, {0xDB3, 0xDBB},
  {0xDBD, 0xDBD}, {0xDC0, 0xDC6}, {0xDD0, 0xDD1}, {0xDD8, 0xDDE}, {0xDE6, 0xDEF}, {0xDF2, 0xDF4},
  {0xE01, 0xE30}, {0xE32, 0xE33}, {0xE3F, 0xE46}, {0xE4F, 0xE5B}, {0xE81, 0xE82}, {0xE84, 0xE84},
  {0xE86, 0xE8A}, {0xE8C, 0xEA3}, {0xEA5, 0xEA5}, {0xEA7, 0xEB0}, {0xEB2, 0xEB3}, {0xEBD, 0xEBD},
  {0xEC0, 0xEC4}, {0xEC6, 0xEC6}, {0xED0, 0xED9}, {0xEDC, 0xEDF}, {0xF00, 0xF17}, {0xF1A, 0xF34},
  {0xF36, 0xF36}, {0xF38, 0xF38}, {0xF3A, 0xF47}, {0xF49, 0xF6C}, {0xF7F, 0xF7F}, {0xF85, 0xF85},
  {0xF88, 0xF8C}, {0xFBE, 0xFC5}, {0xFC7, 0xFCC}, {0xFCE, 0xFDA}, {0x1000, 0x102C},
  {0x1031, 0x1031}, {0x1038, 0x1038}, {0x103B, 0x103C}, {0x103F, 0x1057}, {0x105A, 0x105D},
  {0x1061, 0x1070}, {0x1075, 0x1081}, {0x1083, 0x1084}, {0x1087, 0x108C}, {0x108E, 0x109C},
  {0x109E, 0x10C5}, {0x10C7, 0x10C7}, {0x10CD, 0x10CD}, {0x10D0, 0x1248}, {0x124A, 0x124D},
  {0x1250, 0x1256}, {0x1258, 0x1258}, {0x125A, 0x125D}, {0x1260, 0x1288}, {0x128A, 0x128D},
  {0x1290, 0x12B0}, {0x12B2, 0x12B5}, {0x12B8, 0x12BE}, {0x12C0, 0x12C0}, {0x12C2, 0x12C5},
  {0x12C8, 0x12D6}, {0x12D8, 0x1310}, {0x1312, 0x1315}, {0x1318, 0x135A}, {0x1360, 0x137C},
  {0x1380, 0x1399}, {0x13A0, 0x13F5}, {0x13F8, 0x13FD}, {0x1400, 0x167F}, {0x1681, 0x169C},
  {0x16A0, 0x16F8}, {0x1700, 0x1711}, {0x171F, 0x1731}, {0x1735, 0x1736}, {0x1740, 0x1751},
  {0x1760, 0x176C}, {0x176E, 0x1770}, {0x1780, 0x17B3}, {0x17B6, 0x17B6}, {0x17BE, 0x17C5},
  {0x17C7, 0x17C8}, {0x17D4, 0x17DC}, {0x17E0, 0x17E9}, {0x17F0, 0x17F9}, {0x1800, 0x180A},
  {0x1810, 0x1819}, {0x1820, 0x1878}, {0x1880, 0x1884}, {0x1887, 0x18A8}, {0x18AA, 0x18AA},
  {0x18B0, 0x18F5}, {0x1900, 0x191E}, {0x1923, 0x1926}, {0x1929, 0x192B}, {0x1930, 0x1931},
  {0x1933, 0x1938}, {0x1940, 0x1940}, {0x1944, 0x196D}, {0x1970, 0x1974}, {0x1980, 0x19AB},
  {0x19B0, 0x19C9}, {0x19D0, 0x19DA}, {0x19DE, 0x1A16}, {0x1A19, 0x1A1A}, {0x1A1E, 0x1A55},
  {0x1A57, 0x1A57}, {0x1A61, 0x1A61}, {0x1A63, 0x1A64}, {0x1A6D, 0x1A72}, {0x1A80, 0x1A89},
  {0x1A90, 0x1A99}, {0x1AA0, 0x1AAD}, {0x1B04, 0x1B33}, {0x1B3E, 0x1B41}, {0x1B45, 0x1B4C},
  {0x1B4E, 0x1B6A}, {0x1B74, 0x1B7F}, {0x1B82, 0x1BA1}, {0x1BA6, 0x1BA7}, {0x1BAE, 0x1BE5},
  {0x1BE7, 0x1BE7}, {0x1BEA, 0x1BEC}, {0x1BEE, 0x1BEE}, {0x1BFC, 0x1C2B}, {0x1C34, 0x1C35},
  {0x1C3B, 0x1C49}, {0x1C4D, 0x1C8A}, {0x1C90, 0x1CBA}, {0x1CBD, 0x1CC7}, {0x1CD3, 0x1CD3},
  {0x1CE1, 0x1CE1}, {0x1CE9, 0x1CEC}, {0x1CEE, 0x1CF3}, {0x1CF5, 0x1CF7}, {0x1CFA, 0x1CFA},
  {0x1D00, 0x1DBF}, {0x1E00, 0x1F15}, {0x1F18, 0x1F1D}, {0x1F20, 0x1F45}, {0x1F48, 0x1F4D},
  {0x1F50, 0x1F57}, {0x1F59, 0x1F59}, {0x1F5B, 0x1F5B}, {0x1F5D, 0x1F5D}, {0x1F5F, 0x1F7D},
  {0x1F80, 0x1FB4}, {0x1FB6, 0x1FC4}, {0x1FC6, 0x1FD3}, {0x1FD6, 0x1FDB}, {0x1FDD, 0x1FEF},
  {0x1FF2, 0x1FF4}, {0x1FF6, 0x1FFE}, {0x2010, 0x2027}, {0x2030, 0x205E}, {0x2070, 0x2071},
  {0x2074, 0x208E}, {0x2090, 0x209C}, {0x20A0, 0x20C0}, {0x2100, 0x218B}, {0x2190, 0x2429},
  {0x2440, 0x244A}, {0x2460, 0x2B73}, {0x2B76, 0x2B95}, {0x2B97, 0x2CEE}, {0x2CF2, 0x2CF3},
  {0x2CF9, 0x2D25}, {0x2D27, 0x2D27}, {0x2D2D, 0x2D2D}, {0x2D30, 0x2D67}, {0x2D6F, 0x2D70},
  {0x2D80, 0x2D96}, {0x2DA0, 0x2DA6}, {0x2DA8, 0x2DAE}, {0x2DB0, 0x2DB6}, {0x2DB8, 0x2DBE},
  {0x2DC0, 0x2DC6}, {0x2DC8, 0x2DCE}, {0x2DD0, 0x2DD6}, {0x2DD8, 0x2DDE}, {0x2E00, 0x2E5D},
  {0x2E80, 0x2E99}, {0x2E9B, 0x2EF3}, {0x2F00, 0x2FD5}, {0x2FF0, 0x2FFF}, {0x3001, 0x3029},
  {0x3030, 0x303F}, {0x3041, 0x3096}, {0x309B, 0x30FF}, {0x3105, 0x312F}, {0x3131, 0x318E},
  {0x3190, 0x31E5}, {0x31EF, 0x321E}, {0x3220, 0xA48C}, {0xA490, 0xA4C6}, {0xA4D0, 0xA62B},
  {0xA640, 0xA66E}, {0xA673, 0xA673}, {0xA67E, 0xA69D}, {0xA6A0, 0xA6EF}, {0xA6F2, 0xA6F7},
  {0xA700, 0xA7CD}, {0xA7D0, 0xA7D1}, {0xA7D3, 0xA7D3}, {0xA7D5, 0xA7DC}, {0xA7F2, 0xA801},
  {0xA803, 0xA805}, {0xA807, 0xA80A}, {0xA80C, 0xA824}, {0xA827, 0xA82B}, {0xA830, 0xA839},
  {0xA840, 0xA877}, {0xA880, 0xA8C3}, {0xA8CE, 0xA8D9}, {0xA8F2, 0xA8FE}, {0xA900, 0xA925},
  {0xA92E, 0xA946}, {0xA952, 0xA952}, {0xA95F, 0xA97C}, {0xA983, 0xA9B2}, {0xA9B4, 0xA9B5},
  {0xA9BA, 0xA9BB}, {0xA9BE, 0xA9BF}, {0xA9C1, 0xA9CD}, {0xA9CF, 0xA9D9}, {0xA9DE, 0xA9E4},
  {0xA9E6, 0xA9FE}, {0xAA00, 0xAA28}, {0xAA2F, 0xAA30}, {0xAA33, 0xAA34}, {0xAA40, 0xAA42},
  {0xAA44, 0xAA4B}, {0xAA4D, 0xAA4D}, {0xAA50, 0xAA59}, {0xAA5C, 0xAA7B}, {0xAA7D, 0xAAAF},
  {0xAAB1, 0xAAB1}, {0xAAB5, 0xAAB6}, {0xAAB9, 0xAABD}, {0xAAC0, 0xAAC0}, {0xAAC2, 0xAAC2},
  {0xAADB, 0xAAEB}, {0xAAEE, 0xAAF5}, {0xAB01, 0xAB06}, {0xAB09, 0xAB0E}, {0xAB11, 0xAB16},
  {0xAB20, 0xAB26}, {0xAB28, 0xAB2E}, {0xAB30, 0xAB6B}, {0xAB70, 0xABE4}, {0xABE6, 0xABE7},
  {0xABE9, 0xABEC}, {0xABF0, 0xABF9}, {0xAC00, 0xD7A3}, {0xD7B0, 0xD7C6}, {0xD7CB, 0xD7FB},
  {0xF900, 0xFA6D}, {0xFA70, 0xFAD9}, {0xFB00, 0xFB06}, {0xFB13, 0xFB17}, {0xFB1D, 0xFB1D},
  {0xFB1F, 0xFB36}, {0xFB38, 0xFB3C}, {0xFB3E, 0xFB3E}, {0xFB40, 0xFB41}, {0xFB43, 0xFB44},
  {0xFB46, 0xFBC2}, {0xFBD3, 0xFD8F}, {0xFD92, 0xFDC7}, {0xFDCF, 0xFDCF}, {0xFDF0, 0xFDFF},
  {0xFE10, 0xFE19}, {0xFE30, 0xFE52}, {0xFE54, 0xFE66}, {0xFE68, 0xFE6B}, {0xFE70, 0xFE74},
  {0xFE76, 0xFEFC}, {0xFF01, 0xFF9D}, {0xFFA0, 0xFFBE}, {0xFFC2, 0xFFC7}, {0xFFCA, 0xFFCF},
  {0xFFD2, 0xFFD7}, {0xFFDA, 0xFFDC}, {0xFFE0, 0xFFE6}, {0xFFE8, 0xFFEE}, {0xFFFC, 0xFFFD},
  {0x10000, 0x1000B}, {0x1000D, 0x10026}, {0x10028, 0x1003A}, {0x1003C, 0x1003D},
  {0x1003F, 0x1004D}, {0x10050, 0x1005D}, {0x10080, 0x100FA}, {0x10100, 0x10102},
  {0x10107, 0x10133}, {0x10137, 0x1018E}, {0x10190, 0x1019C}, {0x101A0, 0x101A0},
  {0x101D0, 0x101FC}, {0x10280, 0x1029C}, {0x102A0, 0x102D0}, {0x102E1, 0x102FB},
  {0x10300, 0x10323}, {0x1032D, 0x1034A}, {0x10350, 0x10375}, {0x10380, 0x1039D},
  {0x1039F, 0x103C3}, {0x103C8, 0x103D5}, {0x10400, 0x1049D}, {0x104A0, 0x104A9},
  {0x104B0, 0x104D3}, {0x104D8, 0x104FB}, {0x10500, 0x10527}, {0x10530, 0x10563},
  {0x1056F, 0x1057A}, {0x1057C, 0x1058A}, {0x1058C, 0x10592}, {0x10594, 0x10595},
  {0x10597, 0x105A1}, {0x105A3, 0x105B1}, {0x105B3, 0x105B9}, {0x105BB, 0x105BC},
  {0x105C0, 0x105F3}, {0x10600, 0x10736}, {0x10740, 0x10755}, {0x10760, 0x10767},
  {0x10780, 0x10785}, {0x10787, 0x107B0}, {0x107B2, 0x107BA}, {0x10800, 0x10805},
  {0x10808, 0x10808}, {0x1080A, 0x10835}, {0x10837, 0x10838}, {0x1083C, 0x1083C},
  {0x1083F, 0x10855}, {0x10857, 0x1089E}, {0x108A7, 0x108AF}, {0x108E0, 0x108F2},
  {0x108F4, 0x108F5}, {0x108FB, 0x1091B}, {0x1091F, 0x10939}, {0x1093F, 0x1093F},
  {0x10980, 0x109B7}, {0x109BC, 0x109CF}, {0x109D2, 0x10A00}, {0x10A10, 0x10A13},
  {0x10A15, 0x10A17}, {0x10A19, 0x10A35}, {0x10A40, 0x10A48}, {0x10A50, 0x10A58},
  {0x10A60, 0x10A9F}, {0x10AC0, 0x10AE4}, {0x10AEB, 0x10AF6}, {0x10B00, 0x10B35},
  {0x10B39, 0x10B55}, {0x10B58, 0x10B72}, {0x10B78, 0x10B91}, {0x10B99, 0x10B9C},
  {0x10BA9, 0x10BAF}, {0x10C00, 0x10C48}, {0x10C80, 0x10CB2}, {0x10CC0, 0x10CF2},
  {0x10CFA, 0x10D23}, {0x10D30, 0x10D39}, {0x10D40, 0x10D65}, {0x10D6E, 0x10D85},
  {0x10D8E, 0x10D8F}, {0x10E60, 0x10E7E}, {0x10E80, 0x10EA9}, {0x10EAD, 0x10EAD},
  {0x10EB0, 0x10EB1}, {0x10EC2, 0x10EC4}, {0x10F00, 0x10F27}, {0x10F30, 0x10F45},
  {0x10F51, 0x10F59}, {0x10F70, 0x10F81}, {0x10F86, 0x10F89}, {0x10FB0, 0x10FCB},
  {0x10FE0, 0x10FF6}, {0x11000, 0x11000}, {0x11002, 0x11037}, {0x11047, 0x1104D},
  {0x11052, 0x1106F}, {0x11071, 0x11072}, {0x11075, 0x11075}, {0x11082, 0x110B2},
  {0x110B7, 0x110B8}, {0x110BB, 0x110BC}, {0x110BE, 0x110C1}, {0x110D0, 0x110E8},
  {0x110F0, 0x110F9}, {0x11103, 0x11126}, {0x1112C, 0x1112C}, {0x11136, 0x11147},
  {0x11150, 0x11172}, {0x11174, 0x11176}, {0x11182, 0x111B5}, {0x111BF, 0x111BF},
  {0x111C1, 0x111C8}, {0x111CD, 0x111CE}, {0x111D0, 0x111DF}, {0x111E1, 0x111F4},
  {0x11200, 0x11211}, {0x11213, 0x1122E}, {0x11232, 0x11233}, {0x11238, 0x1123D},
  {0x1123F, 0x11240}, {0x11280, 0x11286}, {0x11288, 0x11288}, {0x1128A, 0x1128D},
  {0x1128F, 0x1129D}, {0x1129F, 0x112A9}, {0x112B0, 0x112DE}, {0x112E0, 0x112E2},
  {0x112F0, 0x112F9}, {0x11302, 0x11303}, {0x11305, 0x1130C}, {0x1130F, 0x11310},
  {0x11313, 0x11328}, {0x1132A, 0x11330}, {0x11332, 0x11333}, {0x11335, 0x11339},
  {0x1133D, 0x1133D}, {0x1133F, 0x1133F}, {0x11341, 0x11344}, {0x11347, 0x11348},
  {0x1134B, 0x1134C}, {0x11350, 0x11350}, {0x1135D, 0x11363}, {0x11380, 0x11389},
  {0x1138B, 0x1138B}, {0x1138E, 0x1138E}, {0x11390, 0x113B5}, {0x113B7, 0x113B7},
  {0x113B9, 0x113BA}, {0x113CA, 0x113CA}, {0x113CC, 0x113CD}, {0x113D1, 0x113D1},
  {0x113D3, 0x113D5}, {0x113D7, 0x113D8}, {0x11400, 0x11437}, {0x11440, 0x11441},
  {0x11445, 0x11445}, {0x11447, 0x1145B}, {0x1145D, 0x1145D}, {0x1145F, 0x11461},
  {0x11480, 0x114AF}, {0x114B1, 0x114B2}, {0x114B9, 0x114B9}, {0x114BB, 0x114BC},
  {0x114BE, 0x114BE}, {0x114C1, 0x114C1}, {0x114C4, 0x114C7}, {0x114D0, 0x114D9},
  {0x11580, 0x115AE}, {0x115B0, 0x115B1}, {0x115B8, 0x115BB}, {0x115BE, 0x115BE},
  {0x115C1, 0x115DB}, {0x11600, 0x11632}, {0x1163B, 0x1163C}, {0x1163E, 0x1163E},
  {0x11641, 0x11644}, {0x11650, 0x11659}, {0x11660, 0x1166C}, {0x11680, 0x116AA},
  {0x116AC, 0x116AC}, {0x116AE, 0x116AF}, {0x116B8, 0x116B9}, {0x116C0, 0x116C9},
  {0x116D0, 0x116E3}, {0x11700, 0x1171A}, {0x1171E, 0x1171E}, {0x11720, 0x11721},
  {0x11726, 0x11726}, {0x11730, 0x11746}, {0x11800, 0x1182E}, {0x11838, 0x11838},
  {0x1183B, 0x1183B}, {0x118A0, 0x118F2}, {0x118FF, 0x11906}, {0x11909, 0x11909},
  {0x1190C, 0x11913}, {0x11915, 0x11916}, {0x11918, 0x1192F}, {0x11931, 0x11935},
  {0x11937, 0x11938}, {0x1193F, 0x11942}, {0x11944, 0x11946}, {0x11950, 0x11959},
  {0x119A0, 0x119A7}, {0x119AA, 0x119D3}, {0x119DC, 0x119DF}, {0x119E1, 0x119E4},
  {0x11A00, 0x11A00}, {0x11A0B, 0x11A32}, {0x11A39, 0x11A3A}, {0x11A3F, 0x11A46},
  {0x11A50, 0x11A50}, {0x11A57, 0x11A58}, {0x11A5C, 0x11A89}, {0x11A97, 0x11A97},
  {0x11A9A, 0x11AA2}, {0x11AB0, 0x11AF8}, {0x11B00, 0x11B09}, {0x11BC0, 0x11BE1},
  {0x11BF0, 0x11BF9}, {0x11C00, 0x11C08}, {0x11C0A, 0x11C2F}, {0x11C3E, 0x11C3E},
  {0x11C40, 0x11C45}, {0x11C50, 0x11C6C}, {0x11C70, 0x11C8F}, {0x11CA9, 0x11CA9},
  {0x11CB1, 0x11CB1}, {0x11CB4, 0x11CB4}, {0x11D00, 0x11D06}, {0x11D08, 0x11D09},
  {0x11D0B, 0x11D30}, {0x11D46, 0x11D46}, {0x11D50, 0x11D59}, {0x11D60, 0x11D65},
  {0x11D67, 0x11D68}, {0x11D6A, 0x11D8E}, {0x11D93, 0x11D94}, {0x11D96, 0x11D96},
  {0x11D98, 0x11D98}, {0x11DA0, 0x11DA9}, {0x11EE0, 0x11EF2}, {0x11EF5, 0x11EF8},
  {0x11F02, 0x11F10}, {0x11F12, 0x11F35}, {0x11F3E, 0x11F3F}, {0x11F43, 0x11F59},
  {0x11FB0, 0x11FB0}, {0x11FC0, 0x11FF1}, {0x11FFF, 0x12399}, {0x12400, 0x1246E},
  {0x12470, 0x12474}, {0x12480, 0x12543}, {0x12F90, 0x12FF2}, {0x13000, 0x1342F},
  {0x13441, 0x13446}, {0x13460, 0x143FA}, {0x14400, 0x14646}, {0x16100, 0x1611D},
  {0x1612A, 0x1612C}, {0x16130, 0x16139}, {0x16800, 0x16A38}, {0x16A40, 0x16A5E},
  {0x16A60, 0x16A69}, {0x16A6E, 0x16ABE}, {0x16AC0, 0x16AC9}, {0x16AD0, 0x16AED},
  {0x16AF5, 0x16AF5}, {0x16B00, 0x16B2F}, {0x16B37, 0x16B45}, {0x16B50, 0x16B59},
  {0x16B5B, 0x16B61}, {0x16B63, 0x16B77}, {0x16B7D, 0x16B8F}, {0x16D40, 0x16D79},
  {0x16E40, 0x16E9A}, {0x16F00, 0x16F4A}, {0x16F50, 0x16F87}, {0x16F93, 0x16F9F},
  {0x16FE0, 0x16FE3}, {0x17000, 0x187F7}, {0x18800, 0x18CD5}, {0x18CFF, 0x18D08},
  {0x1AFF0, 0x1AFF3}, {0x1AFF5, 0x1AFFB}, {0x1AFFD, 0x1AFFE}, {0x1B000, 0x1B122},
  {0x1B132, 0x1B132}, {0x1B150, 0x1B152}, {0x1B155, 0x1B155}, {0x1B164, 0x1B167},
  {0x1B170, 0x1B2FB}, {0x1BC00, 0x1BC6A}, {0x1BC70, 0x1BC7C}, {0x1BC80, 0x1BC88},
  {0x1BC90, 0x1BC99}, {0x1BC9C, 0x1BC9C}, {0x1BC9F, 0x1BC9F}, {0x1CC00, 0x1CCF9},
  {0x1CD00, 0x1CEB3}, {0x1CF50, 0x1CFC3}, {0x1D000, 0x1D0F5}, {0x1D100, 0x1D126},
  {0x1D129, 0x1D164}, {0x1D16A, 0x1D16C}, {0x1D183, 0x1D184}, {0x1D18C, 0x1D1A9},
  {0x1D1AE, 0x1D1EA}, {0x1D200, 0x1D241}, {0x1D245, 0x1D245}, {0x1D2C0, 0x1D2D3},
  {0x1D2E0, 0x1D2F3}, {0x1D300, 0x1D356}, {0x1D360, 0x1D378}, {0x1D400, 0x1D454},
  {0x1D456, 0x1D49C}, {0x1D49E, 0x1D49F}, {0x1D4A2, 0x1D4A2}, {0x1D4A5, 0x1D4A6},
  {0x1D4A9, 0x1D4AC}, {0x1D4AE, 0x1D4B9}, {0x1D4BB, 0x1D4BB}, {0x1D4BD, 0x1D4C3},
  {0x1D4C5, 0x1D505}, {0x1D507, 0x1D50A}, {0x1D50D, 0x1D514}, {0x1D516, 0x1D51C},
  {0x1D51E, 0x1D539}, {0x1D53B, 0x1D53E}, {0x1D540, 0x1D544}, {0x1D546, 0x1D546},
  {0x1D54A, 0x1D550}, {0x1D552, 0x1D6A5}, {0x1D6A8, 0x1D7CB}, {0x1D7CE, 0x1D9FF},
  {0x1DA37, 0x1DA3A}, {0x1DA6D, 0x1DA74}, {0x1DA76, 0x1DA83}, {0x1DA85, 0x1DA8B},
  {0x1DF00, 0x1DF1E}, {0x1DF25, 0x1DF2A}, {0x1E030, 0x1E06D}, {0x1E100, 0x1E12C},
  {0x1E137, 0x1E13D}, {0x1E140, 0x1E149}, {0x1E14E, 0x1E14F}, {0x1E290, 0x1E2AD},
  {0x1E2C0, 0x1E2EB}, {0x1E2F0, 0x1E2F9}, {0x1E2FF, 0x1E2FF}, {0x1E4D0, 0x1E4EB},
  {0x1E4F0, 0x1E4F9}, {0x1E5D0, 0x1E5ED}, {0x1E5F0, 0x1E5FA}, {0x1E5FF, 0x1E5FF},
  {0x1E7E0, 0x1E7E6}, {0x1E7E8, 0x1E7EB}, {0x1E7ED, 0x1E7EE}, {0x1E7F0, 0x1E7FE},
  {0x1E800, 0x1E8C4}, {0x1E8C7, 0x1E8CF}, {0x1E900, 0x1E943}, {0x1E94B, 0x1E94B},
  {0x1E950, 0x1E959}, {0x1E95E, 0x1E95F}, {0x1EC71, 0x1ECB4}, {0x1ED01, 0x1ED3D},
  {0x1EE00, 0x1EE03}, {0x1EE05, 0x1EE1F}, {0x1EE21, 0x1EE22}, {0x1EE24, 0x1EE24},
  {0x1EE27, 0x1EE27}, {0x1EE29, 0x1EE32}, {0x1EE34, 0x1EE37}, {0x1EE39, 0x1EE39},
  {0x1EE3B, 0x1EE3B}, {0x1EE42, 0x1EE42}, {0x1EE47, 0x1EE47}, {0x1EE49, 0x1EE49},
  {0x1EE4B, 0x1EE4B}, {0x1EE4D, 0x1EE4F}, {0x1EE51, 0x1EE52}, {0x1EE54, 0x1EE54},
  {0x1EE57, 0x1EE57}, {0x1EE59, 0x1EE59}, {0x1EE5B, 0x1EE5B}, {0x1EE5D, 0x1EE5D},
  {0x1EE5F, 0x1EE5F}, {0x1EE61, 0x1EE62}, {0x1EE64, 0x1EE64}, {0x1EE67, 0x1EE6A},
  {0x1EE6C, 0x1EE72}, {0x1EE74, 0x1EE77}, {0x1EE79, 0x1EE7C}, {0x1EE7E, 0x1EE7E},
  {0x1EE80, 0x1EE89}, {0x1EE8B, 0x1EE9B}, {0x1EEA1, 0x1EEA3}, {0x1EEA5, 0x1EEA9},
  {0x1EEAB, 0x1EEBB}, {0x1EEF0, 0x1EEF1}, {0x1F000, 0x1F02B}, {0x1F030, 0x1F093},
  {0x1F0A0, 0x1F0AE}, {0x1F0B1, 0x1F0BF}, {0x1F0C1, 0x1F0CF}, {0x1F0D1, 0x1F0F5},
  {0x1F100, 0x1F1AD}, {0x1F1E6, 0x1F202}, {0x1F210, 0x1F23B}, {0x1F240, 0x1F248},
  {0x1F250, 0x1F251}, {0x1F260, 0x1F265}, {0x1F300, 0x1F6D7}, {0x1F6DC, 0x1F6EC},
  {0x1F6F0, 0x1F6FC}, {0x1F700, 0x1F776}, {0x1F77B, 0x1F7D9}, {0x1F7E0, 0x1F7EB},
  {0x1F7F0, 0x1F7F0}, {0x1F800, 0x1F80B}, {0x1F810, 0x1F847}, {0x1F850, 0x1F859},
  {0x1F860, 0x1F887}, {0x1F890, 0x1F8AD}, {0x1F8B0, 0x1F8BB}, {0x1F8C0, 0x1F8C1},
  {0x1F900, 0x1FA53}, {0x1FA60, 0x1FA6D}, {0x1FA70, 0x1FA7C}, {0x1FA80, 0x1FA89},
  {0x1FA8F, 0x1FAC6}, {0x1FACE, 0x1FADC}, {0x1FADF, 0x1FAE9}, {0x1FAF0, 0x1FAF8},
  {0x1FB00, 0x1FB92}, {0x1FB94, 0x1FBF9}, {0x20000, 0x2A6DF}, {0x2A700, 0x2B739},
  {0x2B740, 0x2B81D}, {0x2B820, 0x2CEA1}, {0x2CEB0, 0x2EBE0}, {0x2EBF0, 0x2EE5D},
  {0x2F800, 0x2FA1D}, {0x30000, 0x3134A}, {0x31350, 0x323AF},
};
#define ND_WHITE_SPACE_N 10u
static const uint32_t nd_white_space[ND_WHITE_SPACE_N][2] = {
  {0x9, 0xD}, {0x20, 0x20}, {0x85, 0x85}, {0xA0, 0xA0}, {0x1680, 0x1680}, {0x2000, 0x200A},
  {0x2028, 0x2029}, {0x202F, 0x202F}, {0x205F, 0x205F}, {0x3000, 0x3000},
};

static int in_ranges(const uint32_t (*r)[2], size_t n, uint32_t c) {
  size_t lo = 0, hi = n;
  while (lo < hi) {
    size_t mid = lo + (hi - lo) / 2;
    if (c < r[mid][0])
      hi = mid;
    else if (c > r[mid][1])
      lo = mid + 1;
    else
      return 1;
  }
  return 0;
}

int nd_rust_is_whitespace(uint32_t c) {
  return in_ranges(nd_white_space, ND_WHITE_SPACE_N, c);
}

/* Decode the UTF-8 char at s[i] (valid input assumed; an invalid byte decodes as itself, one
 * byte long, so a caller handed bad input never reads out of bounds). */
static uint32_t u8_next(const unsigned char *s, size_t n, size_t i, size_t *clen) {
  uint32_t c = s[i];
  size_t k, l;
  if (c < 0x80u) {
    *clen = 1;
    return c;
  }
  if (c >= 0xF0u)
    l = 4;
  else if (c >= 0xE0u)
    l = 3;
  else if (c >= 0xC0u)
    l = 2;
  else
    l = 1;
  if (l == 1 || n - i < l) {
    *clen = 1;
    return c;
  }
  c &= (l == 2) ? 0x1Fu : (l == 3) ? 0x0Fu : 0x07u;
  for (k = 1; k < l; k++) {
    if ((s[i + k] & 0xC0u) != 0x80u) {
      *clen = 1;
      return s[i];
    }
    c = (c << 6) | (s[i + k] & 0x3Fu);
  }
  *clen = l;
  return c;
}

/* The start of the char that ends at s[end-1]. */
static size_t u8_prev(const unsigned char *s, size_t end) {
  size_t i = end - 1, k = 0;
  while (i > 0 && k < 3 && (s[i] & 0xC0u) == 0x80u) {
    i--;
    k++;
  }
  {
    size_t l;
    (void)u8_next(s, end, i, &l);
    if (i + l != end) return end - 1;
  }
  return i;
}

void nd_rust_trim_end(const char *s, size_t len, size_t *end) {
  const unsigned char *u = (const unsigned char *)s;
  size_t e = len;
  while (e > 0) {
    size_t st = u8_prev(u, e), l;
    uint32_t c = u8_next(u, e, st, &l);
    if (!nd_rust_is_whitespace(c)) break;
    e = st;
  }
  *end = e;
}

void nd_rust_trim(const char *s, size_t len, size_t *start, size_t *end) {
  const unsigned char *u = (const unsigned char *)s;
  size_t b = 0, e;
  while (b < len) {
    size_t l;
    uint32_t c = u8_next(u, len, b, &l);
    if (!nd_rust_is_whitespace(c)) break;
    b += l;
  }
  if (b == len) {
    /* Nothing kept: Rust's trim_matches returns the empty slice at the start. */
    *start = *end = 0;
    return;
  }
  nd_rust_trim_end(s + b, len - b, &e);
  *start = b;
  *end = b + e;
}

/* ─── format!("{:?}", s) ───────────────────────────────────────────────────────────────────── */

static void w_fmt(nd_json_writer *w, const char *fmt, ...) {
  char buf[96];
  va_list ap;
  int n;
  va_start(ap, fmt);
  n = vsnprintf(buf, sizeof buf, fmt, ap);
  va_end(ap);
  if (n < 0 || (size_t)n >= sizeof buf) {
    if (!w->err) w->err = ND_E_ARG;
    return;
  }
  nd_json_writer_raw(w, buf, (size_t)n);
}

static void w_str(nd_json_writer *w, const char *s) { nd_json_writer_raw(w, s, strlen(s)); }

void nd_rust_debug_str(nd_json_writer *w, const char *s, size_t len) {
  const unsigned char *u = (const unsigned char *)s;
  size_t i = 0, run = 0;
  nd_json_writer_raw(w, "\"", 1);
  while (i < len) {
    size_t l;
    uint32_t c = u8_next(u, len, i, &l);
    const char *esc = NULL;
    int hex = 0;
    if (c < 0x80u) {
      switch (c) {
        case 0: esc = "\\0"; break;
        case '\t': esc = "\\t"; break;
        case '\r': esc = "\\r"; break;
        case '\n': esc = "\\n"; break;
        case '\\': esc = "\\\\"; break;
        case '"': esc = "\\\""; break;
        default: hex = (c < 0x20u || c == 0x7Fu); break;
      }
    } else {
      hex = !in_ranges(nd_dbg_printable, ND_DBG_PRINTABLE_N, c);
    }
    if (esc || hex) {
      if (i > run) nd_json_writer_raw(w, s + run, i - run);
      if (esc)
        w_str(w, esc);
      else
        w_fmt(w, "\\u{%lx}", (unsigned long)c);
      run = i + l;
    }
    i += l;
  }
  if (len > run) nd_json_writer_raw(w, s + run, len - run);
  nd_json_writer_raw(w, "\"", 1);
}

/* ─── serde_json::from_str::<Value> error text ─────────────────────────────────────────────────
 * nd_json decides acceptance; when it refuses, this transcription of serde_json 1.0.149's
 * Deserializer (de.rs, read.rs: SliceRead under StrRead, validating strings, no float_roundtrip)
 * finds the error serde_json reports and where: `error` positions at the read index, `peek_error`
 * one byte further (capped at the input length). */

typedef struct {
  const unsigned char *s;
  size_t n, i;
  int remaining_depth;
  const char *code;
  size_t at;
} sj;

static int sj_err(sj *p, const char *code) {
  p->code = code;
  p->at = p->i;
  return 1;
}
static int sj_peek_err(sj *p, const char *code) {
  p->code = code;
  p->at = p->i + 1 < p->n ? p->i + 1 : p->n;
  return 1;
}
static int sj_peek(sj *p) { return p->i < p->n ? (int)p->s[p->i] : -1; }
static int sj_peek_or_null(sj *p) { return p->i < p->n ? (int)p->s[p->i] : 0; }
static int sj_ws(sj *p) {
  while (p->i < p->n) {
    unsigned char c = p->s[p->i];
    if (c == ' ' || c == '\n' || c == '\t' || c == '\r')
      p->i++;
    else
      break;
  }
  return sj_peek(p);
}
static int sj_digit(int c) { return c >= '0' && c <= '9'; }

#define SJ_EOF_LIST "EOF while parsing a list"
#define SJ_EOF_OBJECT "EOF while parsing an object"
#define SJ_EOF_STRING "EOF while parsing a string"
#define SJ_EOF_VALUE "EOF while parsing a value"
#define SJ_RANGE "number out of range"
#define SJ_INVALID_NUMBER "invalid number"
#define SJ_INVALID_ESCAPE "invalid escape"

static int sj_value(sj *p);

static int sj_from_parts(sj *p, uint64_t significand, int64_t exponent) {
  /* Only an exponent >= 0 can fail: f *= 10^exponent overflowing, or no table entry. */
  if (exponent < 0) return 0;
  if (exponent > 308) return significand != 0 ? sj_err(p, SJ_RANGE) : 0;
  {
    char buf[16];
    double pow, f;
    snprintf(buf, sizeof buf, "1e%d", (int)exponent);
    pow = strtod(buf, NULL); /* the correctly rounded literal, as serde_json's POW10 table */
    f = (double)significand * pow;
    if (f > DBL_MAX) return sj_err(p, SJ_RANGE);
  }
  return 0;
}

static int sj_exponent(sj *p, uint64_t significand, int64_t starting_exp) {
  int positive_exp = 1, c;
  int32_t exp;
  int64_t final_exp;
  p->i++;
  c = sj_peek_or_null(p);
  if (c == '+') {
    p->i++;
  } else if (c == '-') {
    p->i++;
    positive_exp = 0;
  }
  if (p->i >= p->n) return sj_err(p, SJ_EOF_VALUE);
  c = p->s[p->i++];
  if (!sj_digit(c)) return sj_err(p, SJ_INVALID_NUMBER);
  exp = c - '0';
  while (sj_digit(c = sj_peek_or_null(p))) {
    int32_t digit = c - '0';
    p->i++;
    if (exp >= INT32_MAX / 10 && (exp > INT32_MAX / 10 || digit > INT32_MAX % 10)) {
      if (significand != 0 && positive_exp) return sj_err(p, SJ_RANGE);
      while (sj_digit(sj_peek_or_null(p))) p->i++;
      return 0;
    }
    exp = exp * 10 + digit;
  }
  final_exp = positive_exp ? starting_exp + exp : starting_exp - exp;
  if (final_exp > INT32_MAX) final_exp = INT32_MAX;
  if (final_exp < INT32_MIN) final_exp = INT32_MIN;
  return sj_from_parts(p, significand, final_exp);
}

static int sj_ovf_u64(uint64_t a, uint64_t b) {
  return a >= UINT64_MAX / 10u && (a > UINT64_MAX / 10u || b > UINT64_MAX % 10u);
}

static int sj_decimal(sj *p, uint64_t significand, int64_t exp_before) {
  int64_t exp_after = 0;
  int c;
  p->i++;
  while (sj_digit(c = sj_peek_or_null(p))) {
    uint64_t digit = (uint64_t)(c - '0');
    if (sj_ovf_u64(significand, digit)) {
      int64_t exponent = exp_before + exp_after;
      while (sj_digit(sj_peek_or_null(p))) p->i++;
      c = sj_peek_or_null(p);
      if (c == 'e' || c == 'E') return sj_exponent(p, significand, exponent);
      return sj_from_parts(p, significand, exponent);
    }
    p->i++;
    significand = significand * 10u + digit;
    exp_after -= 1;
  }
  if (exp_after == 0) {
    if (p->i < p->n) return sj_peek_err(p, SJ_INVALID_NUMBER);
    return sj_peek_err(p, SJ_EOF_VALUE);
  }
  c = sj_peek_or_null(p);
  if (c == 'e' || c == 'E') return sj_exponent(p, significand, exp_before + exp_after);
  return sj_from_parts(p, significand, exp_before + exp_after);
}

static int sj_number(sj *p, uint64_t significand) {
  int c = sj_peek_or_null(p);
  if (c == '.') return sj_decimal(p, significand, 0);
  if (c == 'e' || c == 'E') return sj_exponent(p, significand, 0);
  return 0;
}

static int sj_integer(sj *p) {
  int c;
  if (p->i >= p->n) return sj_err(p, SJ_EOF_VALUE);
  c = p->s[p->i++];
  if (c == '0') {
    if (sj_digit(sj_peek_or_null(p))) return sj_peek_err(p, SJ_INVALID_NUMBER);
    return sj_number(p, 0);
  }
  if (c >= '1' && c <= '9') {
    uint64_t significand = (uint64_t)(c - '0');
    for (;;) {
      c = sj_peek_or_null(p);
      if (sj_digit(c)) {
        uint64_t digit = (uint64_t)(c - '0');
        if (sj_ovf_u64(significand, digit)) {
          int64_t exponent = 0;
          for (;;) {
            c = sj_peek_or_null(p);
            if (sj_digit(c)) {
              p->i++;
              exponent += 1;
            } else if (c == '.') {
              return sj_decimal(p, significand, exponent);
            } else if (c == 'e' || c == 'E') {
              return sj_exponent(p, significand, exponent);
            } else {
              return sj_from_parts(p, significand, exponent);
            }
          }
        }
        p->i++;
        significand = significand * 10u + digit;
      } else {
        return sj_number(p, significand);
      }
    }
  }
  return sj_err(p, SJ_INVALID_NUMBER);
}

static int sj_hex4(sj *p, uint32_t *out) {
  uint32_t v = 0;
  int k;
  if (p->n - p->i < 4) {
    p->i = p->n;
    return sj_err(p, SJ_EOF_STRING);
  }
  for (k = 0; k < 4; k++) {
    int c = p->s[p->i + (size_t)k], h;
    if (c >= '0' && c <= '9')
      h = c - '0';
    else if (c >= 'a' && c <= 'f')
      h = c - 'a' + 10;
    else if (c >= 'A' && c <= 'F')
      h = c - 'A' + 10;
    else
      h = -1;
    if (h < 0) {
      p->i += 4;
      return sj_err(p, SJ_INVALID_ESCAPE);
    }
    v = v * 16u + (uint32_t)h;
  }
  p->i += 4;
  *out = v;
  return 0;
}

/* After the opening quote. */
static int sj_string(sj *p) {
  for (;;) {
    unsigned char c;
    while (p->i < p->n) {
      c = p->s[p->i];
      if (c == '"' || c == '\\' || c < 0x20u) break;
      p->i++;
    }
    if (p->i == p->n) return sj_err(p, SJ_EOF_STRING);
    c = p->s[p->i];
    if (c == '"') {
      p->i++;
      return 0;
    }
    if (c != '\\') {
      p->i++;
      return sj_err(p, "control character (\\u0000-\\u001F) found while parsing a string");
    }
    p->i++;
    if (p->i >= p->n) return sj_err(p, SJ_EOF_STRING);
    c = p->s[p->i++];
    switch (c) {
      case '"': case '\\': case '/': case 'b': case 'f': case 'n': case 'r': case 't': break;
      case 'u': {
        uint32_t v, v2;
        if (sj_hex4(p, &v)) return 1;
        if (v >= 0xDC00u && v <= 0xDFFFu) return sj_err(p, "lone leading surrogate in hex escape");
        if (v < 0xD800u || v > 0xDBFFu) break;
        if (p->i >= p->n) return sj_err(p, SJ_EOF_STRING);
        if (p->s[p->i] != '\\') {
          p->i++;
          return sj_err(p, "unexpected end of hex escape");
        }
        p->i++;
        if (p->i >= p->n) return sj_err(p, SJ_EOF_STRING);
        if (p->s[p->i] != 'u') {
          p->i++;
          return sj_err(p, "unexpected end of hex escape");
        }
        p->i++;
        if (sj_hex4(p, &v2)) return 1;
        if (v2 < 0xDC00u || v2 > 0xDFFFu) return sj_err(p, "lone leading surrogate in hex escape");
        break;
      }
      default:
        return sj_err(p, SJ_INVALID_ESCAPE);
    }
  }
}

static int sj_ident(sj *p, const char *rest) {
  for (; *rest; rest++) {
    if (p->i >= p->n) return sj_err(p, SJ_EOF_VALUE);
    if (p->s[p->i++] != (unsigned char)*rest) return sj_err(p, "expected ident");
  }
  return 0;
}

static int sj_array(sj *p) {
  int first = 1, c;
  p->remaining_depth--;
  if (p->remaining_depth == 0) return sj_peek_err(p, "recursion limit exceeded");
  p->i++;
  for (;;) {
    c = sj_ws(p);
    if (c < 0) return sj_peek_err(p, SJ_EOF_LIST);
    if (c == ']') break;
    if (first) {
      first = 0;
    } else if (c == ',') {
      p->i++;
      c = sj_ws(p);
      if (c == ']') return sj_peek_err(p, "trailing comma");
      if (c < 0) return sj_peek_err(p, SJ_EOF_VALUE);
    } else {
      return sj_peek_err(p, "expected `,` or `]`");
    }
    if (sj_value(p)) return 1;
  }
  p->i++;
  p->remaining_depth++;
  return 0;
}

static int sj_object(sj *p) {
  int first = 1, c;
  p->remaining_depth--;
  if (p->remaining_depth == 0) return sj_peek_err(p, "recursion limit exceeded");
  p->i++;
  for (;;) {
    c = sj_ws(p);
    if (c < 0) return sj_peek_err(p, SJ_EOF_OBJECT);
    if (c == '}') break;
    if (first) {
      first = 0;
      if (c != '"') return sj_peek_err(p, "key must be a string");
    } else if (c == ',') {
      p->i++;
      c = sj_ws(p);
      if (c == '}') return sj_peek_err(p, "trailing comma");
      if (c < 0) return sj_peek_err(p, SJ_EOF_VALUE);
      if (c != '"') return sj_peek_err(p, "key must be a string");
    } else {
      return sj_peek_err(p, "expected `,` or `}`");
    }
    p->i++;
    if (sj_string(p)) return 1;
    c = sj_ws(p);
    if (c < 0) return sj_peek_err(p, SJ_EOF_OBJECT);
    if (c != ':') return sj_peek_err(p, "expected `:`");
    p->i++;
    if (sj_value(p)) return 1;
  }
  p->i++;
  p->remaining_depth++;
  return 0;
}

static int sj_value(sj *p) {
  int c = sj_ws(p);
  if (c < 0) return sj_peek_err(p, SJ_EOF_VALUE);
  switch (c) {
    case 'n': p->i++; return sj_ident(p, "ull");
    case 't': p->i++; return sj_ident(p, "rue");
    case 'f': p->i++; return sj_ident(p, "alse");
    case '-': p->i++; return sj_integer(p);
    case '"': p->i++; return sj_string(p);
    case '[': return sj_array(p);
    case '{': return sj_object(p);
    default:
      if (sj_digit(c)) return sj_integer(p);
      return sj_peek_err(p, "expected value");
  }
}

/* 1 and p->code/p->at set when serde_json rejects the text. */
static int sj_check(sj *p) {
  if (sj_value(p)) return 1;
  if (sj_ws(p) >= 0) return sj_peek_err(p, "trailing characters");
  return 0;
}

static char *dup_cstr(const char *s) {
  size_t n = strlen(s);
  char *d = (char *)malloc(n + 1);
  if (d) memcpy(d, s, n + 1);
  return d;
}

/* Rust's `{}` of a usize. */
#define ZU(x) ((unsigned long long)(x))

int nd_serde_parse(const char *text, size_t len, nd_json_doc **doc, char **errmsg) {
  nd_json_limits lim;
  nd_err err;
  int rc;
  sj p;
  if (errmsg) *errmsg = NULL;
  if (!doc || (!text && len)) return ND_E_ARG;
  *doc = NULL;
  lim.max_input = len;
  lim.max_depth = ND_JSON_SERDE_MAX_DEPTH;
  lim.max_elements = len + 1; /* every value and key takes at least one byte of input */
  err.msg[0] = '\0';
  rc = nd_json_parse(text ? text : "", len, &lim, doc, &err);
  if (rc == ND_OK) return ND_OK;
  if (rc == ND_E_NOMEM) return ND_E_NOMEM;
  memset(&p, 0, sizeof p);
  p.s = (const unsigned char *)(text ? text : "");
  p.n = len;
  p.remaining_depth = 128;
  if (sj_check(&p)) {
    size_t start = 0, line = 1, k;
    for (k = 0; k < p.at; k++)
      if (p.s[k] == '\n') {
        line++;
        start = k + 1;
      }
    if (errmsg) {
      nd_json_writer w;
      nd_json_writer_init(&w, 0);
      w_str(&w, p.code);
      w_fmt(&w, " at line %llu column %llu", ZU(line), ZU(p.at - start));
      if (w.err) {
        nd_json_writer_free(&w);
        return ND_E_NOMEM;
      }
      *errmsg = w.data;
    }
    return ND_E_FORMAT;
  }
  /* nd_json refused what serde_json accepts: report it rather than guess. */
  if (errmsg) {
    nd_json_writer w;
    nd_json_writer_init(&w, 0);
    w_str(&w, "internal: nd_json disagrees with serde_json: ");
    w_str(&w, err.msg);
    if (w.err) {
      nd_json_writer_free(&w);
      return ND_E_NOMEM;
    }
    *errmsg = w.data;
  }
  return ND_E_FORMAT;
}

/* ─── serde_json Map view ──────────────────────────────────────────────────────────────────── */

static int key_cmp(const nd_json_member *a, const nd_json_member *b) {
  size_t n = a->key_len < b->key_len ? a->key_len : b->key_len;
  int c = n ? memcmp(a->key, b->key, n) : 0;
  if (c) return c;
  return a->key_len < b->key_len ? -1 : a->key_len > b->key_len ? 1 : 0;
}

int nd_sorted_members(const nd_json_value *obj, const nd_json_member ***out, size_t *n) {
  size_t len, k, width, kept;
  const nd_json_member **a, **t;
  if (!obj || obj->type != ND_JSON_OBJECT || !out || !n) return ND_E_ARG;
  len = obj->u.obj.len;
  *out = NULL;
  *n = 0;
  if (len == 0) return ND_OK;
  a = (const nd_json_member **)nd_calloc(len, sizeof *a);
  t = (const nd_json_member **)nd_calloc(len, sizeof *t);
  if (!a || !t) {
    free(a);
    free(t);
    return ND_E_NOMEM;
  }
  for (k = 0; k < len; k++) a[k] = &obj->u.obj.members[k];
  /* Stable merge sort: duplicates stay in source order, so the last of a run is the kept one. */
  for (width = 1; width < len; width *= 2) {
    size_t lo;
    for (lo = 0; lo < len; lo += 2 * width) {
      size_t mid = lo + width < len ? lo + width : len;
      size_t hi = lo + 2 * width < len ? lo + 2 * width : len;
      size_t x = lo, y = mid, o = lo;
      while (x < mid && y < hi) {
        if (key_cmp(a[y], a[x]) < 0)
          t[o++] = a[y++];
        else
          t[o++] = a[x++];
      }
      while (x < mid) t[o++] = a[x++];
      while (y < hi) t[o++] = a[y++];
    }
    memcpy(a, t, len * sizeof *a);
  }
  kept = 0;
  for (k = 0; k < len; k++) {
    if (k + 1 < len && key_cmp(a[k], a[k + 1]) == 0) continue;
    a[kept++] = a[k];
  }
  free(t);
  *out = a;
  *n = kept;
  return ND_OK;
}

/* ─── Catalogue ────────────────────────────────────────────────────────────────────────────── */

static int bytes_eq(const char *a, size_t al, const char *b, size_t bl) {
  return al == bl && (al == 0 || memcmp(a, b, al) == 0);
}
static int eq_cstr(const char *a, size_t al, const char *lit) {
  return bytes_eq(a, al, lit, strlen(lit));
}

static char *dup_bytes(const char *s, size_t n) {
  char *d;
  if (n == (size_t)-1) return NULL;
  d = (char *)malloc(n + 1);
  if (!d) return NULL;
  if (n) memcpy(d, s, n);
  d[n] = '\0';
  return d;
}

/* Errors are written into `e` (the caller's prefix already there); returns ND_E_FORMAT, or
 * ND_E_NOMEM when the writer ran out. */
static int fail(nd_json_writer *e, const char *msg) {
  w_str(e, msg);
  return e->err ? ND_E_NOMEM : ND_E_FORMAT;
}

static int only_keys(const nd_json_member *const *mv, size_t n, const char *const *allowed,
                     const char *what, nd_json_writer *e) {
  size_t k;
  for (k = 0; k < n; k++) {
    const char *const *a;
    int ok = 0;
    for (a = allowed; *a; a++)
      if (eq_cstr(mv[k]->key, mv[k]->key_len, *a)) ok = 1;
    if (!ok) {
      w_str(e, what);
      w_str(e, ": unsupported keyword ");
      nd_rust_debug_str(e, mv[k]->key, mv[k]->key_len);
      return e->err ? ND_E_NOMEM : ND_E_FORMAT;
    }
  }
  return ND_OK;
}

static int obj_view(const nd_json_value *v, const nd_json_member ***mv, size_t *n) {
  int rc = nd_sorted_members(v, mv, n);
  return rc == ND_OK ? ND_OK : ND_E_NOMEM;
}

static void param_free(nd_param *p) {
  size_t k;
  free(p->name);
  for (k = 0; k < p->n_enum; k++) free(p->enum_values[k]);
  free(p->enum_values);
  free(p->enum_lens);
}

static void tool_free(nd_tool *t) {
  size_t k;
  free(t->name);
  free(t->snake_name);
  for (k = 0; k < t->n_params; k++) param_free(&t->params[k]);
  free(t->params);
  for (k = 0; k < t->n_required; k++) free(t->required[k]);
  free(t->required);
  free(t->required_lens);
  free(t->json);
  memset(t, 0, sizeof *t);
}

void nd_catalogue_free(nd_catalogue *c) {
  size_t k;
  if (!c) return;
  for (k = 0; k < c->n_tools; k++) tool_free(&c->tools[k]);
  free(c->tools);
  free(c);
}

/* The closures `int` / `num` of parse_param. 1 = present and stored, 0 = absent, <0 = error. */
static int get_int(const nd_json_value *m, const char *k, int64_t *out, nd_json_writer *e) {
  const nd_json_value *x = nd_json_get_cstr(m, k);
  if (!x) return 0;
  if (!nd_json_as_i64(x, out)) {
    w_str(e, k);
    return fail(e, " must be an integer");
  }
  return 1;
}
static int get_num(const nd_json_value *m, const char *k, double *out, nd_json_writer *e) {
  const nd_json_value *x = nd_json_get_cstr(m, k);
  if (!x) return 0;
  if (!nd_json_as_f64(x, out)) {
    w_str(e, k);
    return fail(e, " must be a number");
  }
  return 1;
}

static int parse_param(const nd_json_value *v, nd_param *p, nd_json_writer *e) {
  static const char *const k_string[] = {"type", "description", "enum", "maxLength", NULL};
  static const char *const k_integer[] = {"type", "description", "minimum", "maximum", NULL};
  static const char *const k_boolean[] = {"type", "description", NULL};
  const nd_json_value *tv;
  const nd_json_member **mv = NULL;
  size_t nm = 0;
  const char *ty;
  size_t tl;
  int rc;
  if (v->type != ND_JSON_OBJECT) return fail(e, "not an object");
  tv = nd_json_get_cstr(v, "type");
  if (!tv || !nd_json_as_str(tv, &ty, &tl)) return fail(e, "missing string type");
  if (obj_view(v, &mv, &nm)) return ND_E_NOMEM;
  if (eq_cstr(ty, tl, "string")) {
    const nd_json_value *ev;
    int64_t ml;
    p->kind = ND_PARAM_STRING;
    if ((rc = only_keys(mv, nm, k_string, "string", e))) goto out;
    ev = nd_json_get_cstr(v, "enum");
    if (ev) {
      size_t k, n;
      if (ev->type != ND_JSON_ARRAY) {
        rc = fail(e, "enum must be an array");
        goto out;
      }
      n = ev->u.arr.len;
      if (n == 0) {
        rc = fail(e, "enum is empty");
        goto out;
      }
      for (k = 0; k < n; k++)
        if (ev->u.arr.items[k].type != ND_JSON_STRING) {
          rc = fail(e, "enum values must be strings");
          goto out;
        }
      p->enum_values = (char **)nd_calloc(n, sizeof *p->enum_values);
      p->enum_lens = (size_t *)nd_calloc(n, sizeof *p->enum_lens);
      if (!p->enum_values || !p->enum_lens) {
        rc = ND_E_NOMEM;
        goto out;
      }
      p->has_enum = 1;
      for (k = 0; k < n; k++) {
        const nd_json_value *x = &ev->u.arr.items[k];
        p->enum_values[k] = dup_bytes(x->u.str.ptr, x->u.str.len);
        if (!p->enum_values[k]) {
          rc = ND_E_NOMEM;
          goto out;
        }
        p->enum_lens[k] = x->u.str.len;
        p->n_enum = k + 1;
      }
    }
    rc = get_int(v, "maxLength", &ml, e);
    if (rc < 0) goto out;
    if (rc == 1) {
      /* usize::try_from(i64) */
      if (ml < 0 || (uint64_t)ml > (uint64_t)SIZE_MAX) {
        rc = fail(e, "maxLength must be >= 0");
        goto out;
      }
      p->has_max_length = 1;
      p->max_length = (size_t)ml;
    }
    rc = ND_OK;
  } else if (eq_cstr(ty, tl, "integer")) {
    p->kind = ND_PARAM_INTEGER;
    if ((rc = only_keys(mv, nm, k_integer, "integer", e))) goto out;
    rc = get_int(v, "minimum", &p->imin, e);
    if (rc < 0) goto out;
    p->has_imin = rc;
    rc = get_int(v, "maximum", &p->imax, e);
    if (rc < 0) goto out;
    p->has_imax = rc;
    rc = ND_OK;
  } else if (eq_cstr(ty, tl, "number")) {
    p->kind = ND_PARAM_NUMBER;
    if ((rc = only_keys(mv, nm, k_integer, "number", e))) goto out;
    rc = get_num(v, "minimum", &p->fmin, e);
    if (rc < 0) goto out;
    p->has_fmin = rc;
    rc = get_num(v, "maximum", &p->fmax, e);
    if (rc < 0) goto out;
    p->has_fmax = rc;
    rc = ND_OK;
  } else if (eq_cstr(ty, tl, "boolean")) {
    p->kind = ND_PARAM_BOOLEAN;
    rc = only_keys(mv, nm, k_boolean, "boolean", e);
  } else {
    w_str(e, "type ");
    nd_rust_debug_str(e, ty, tl);
    rc = fail(e, " is not supported (string, integer, number, boolean)");
  }
out:
  free(mv);
  return rc;
}

static int is_name_byte(unsigned char b) {
  return (b >= 'A' && b <= 'Z') || (b >= 'a' && b <= 'z') || (b >= '0' && b <= '9') || b == '_' ||
         b == '-';
}

static int parse_tool(const nd_json_value *t, const char *raw, size_t raw_len, nd_tool *tool,
                      nd_json_writer *e) {
  static const char *const k_tool[] = {"name", "description", "parameters", NULL};
  static const char *const k_params[] = {"type", "properties", "required", NULL};
  const nd_json_member **mv = NULL, **pv = NULL, **props = NULL;
  size_t nm = 0, np = 0, nprops = 0, k;
  const nd_json_value *nv, *d, *pm;
  const char *name;
  size_t name_len;
  int rc = ND_OK;
  if (t->type != ND_JSON_OBJECT) return fail(e, "not an object");
  if (obj_view(t, &mv, &nm)) return ND_E_NOMEM;
  if ((rc = only_keys(mv, nm, k_tool, "tool", e))) goto out;
  nv = nd_json_get_cstr(t, "name");
  if (!nv || !nd_json_as_str(nv, &name, &name_len)) {
    rc = fail(e, "missing string name");
    goto out;
  }
  {
    int ok = name_len >= 1 && name_len <= 64;
    for (k = 0; ok && k < name_len; k++) ok = is_name_byte((unsigned char)name[k]);
    if (!ok) {
      w_str(e, "name ");
      nd_rust_debug_str(e, name, name_len);
      rc = fail(e, " must be 1..=64 of [A-Za-z0-9_-]");
      goto out;
    }
  }
  d = nd_json_get_cstr(t, "description");
  if (d && d->type != ND_JSON_STRING) {
    rc = fail(e, "description must be a string");
    goto out;
  }
  pm = nd_json_get_cstr(t, "parameters");
  if (pm) {
    const nd_json_value *ty, *pr, *rq;
    const char *ts;
    size_t tl;
    if (pm->type != ND_JSON_OBJECT) {
      rc = fail(e, "parameters must be an object");
      goto out;
    }
    if (obj_view(pm, &pv, &np)) {
      rc = ND_E_NOMEM;
      goto out;
    }
    if ((rc = only_keys(pv, np, k_params, "parameters", e))) goto out;
    ty = nd_json_get_cstr(pm, "type");
    if (!ty || !nd_json_as_str(ty, &ts, &tl) || !eq_cstr(ts, tl, "object")) {
      rc = fail(e, "parameters.type must be \"object\"");
      goto out;
    }
    pr = nd_json_get_cstr(pm, "properties");
    if (pr) {
      if (pr->type != ND_JSON_OBJECT) {
        rc = fail(e, "properties must be an object");
        goto out;
      }
      if (obj_view(pr, &props, &nprops)) {
        rc = ND_E_NOMEM;
        goto out;
      }
      if (nprops > ND_MAX_PARAMS) {
        w_fmt(e, "more than %u parameters", ND_MAX_PARAMS);
        rc = e->err ? ND_E_NOMEM : ND_E_FORMAT;
        goto out;
      }
      tool->params = (nd_param *)nd_calloc(nprops, sizeof *tool->params);
      if (!tool->params) {
        rc = ND_E_NOMEM;
        goto out;
      }
      for (k = 0; k < nprops; k++) {
        nd_param *p = &tool->params[k];
        tool->n_params = k + 1;
        p->name = dup_bytes(props[k]->key, props[k]->key_len);
        if (!p->name) {
          rc = ND_E_NOMEM;
          goto out;
        }
        p->name_len = props[k]->key_len;
        w_str(e, "parameter ");
        nd_rust_debug_str(e, props[k]->key, props[k]->key_len);
        w_str(e, ": ");
        if (e->err) {
          rc = ND_E_NOMEM;
          goto out;
        }
        if ((rc = parse_param(&props[k]->value, p, e))) goto out;
        nd_json_writer_reset(e); /* drop the "parameter" prefix (the "tool i" one is re-added) */
      }
    }
    rq = nd_json_get_cstr(pm, "required");
    if (rq) {
      size_t n;
      if (rq->type != ND_JSON_ARRAY) {
        rc = fail(e, "required must be an array");
        goto out;
      }
      n = rq->u.arr.len;
      tool->required = (char **)nd_calloc(n, sizeof *tool->required);
      tool->required_lens = (size_t *)nd_calloc(n, sizeof *tool->required_lens);
      if (!tool->required || !tool->required_lens) {
        rc = ND_E_NOMEM;
        goto out;
      }
      for (k = 0; k < n; k++) {
        const nd_json_value *x = &rq->u.arr.items[k];
        size_t j;
        int declared = 0, dup = 0;
        if (x->type != ND_JSON_STRING) {
          rc = fail(e, "required entries must be strings");
          goto out;
        }
        for (j = 0; j < tool->n_params; j++)
          if (bytes_eq(tool->params[j].name, tool->params[j].name_len, x->u.str.ptr, x->u.str.len))
            declared = 1;
        if (!declared) {
          w_str(e, "required ");
          nd_rust_debug_str(e, x->u.str.ptr, x->u.str.len);
          rc = fail(e, " is not a declared property");
          goto out;
        }
        for (j = 0; j < tool->n_required; j++)
          if (bytes_eq(tool->required[j], tool->required_lens[j], x->u.str.ptr, x->u.str.len))
            dup = 1;
        if (!dup) {
          char *s = dup_bytes(x->u.str.ptr, x->u.str.len);
          if (!s) {
            rc = ND_E_NOMEM;
            goto out;
          }
          tool->required[tool->n_required] = s;
          tool->required_lens[tool->n_required] = x->u.str.len;
          tool->n_required++;
        }
      }
    }
  }
  tool->json = dup_bytes(raw, raw_len);
  tool->name = dup_bytes(name, name_len);
  tool->snake_name = tool->name ? nd_to_snake_case(tool->name) : NULL;
  if (!tool->json || !tool->name || !tool->snake_name) {
    rc = ND_E_NOMEM;
    goto out;
  }
  tool->json_len = raw_len;
  tool->name_len = name_len;
  tool->snake_len = strlen(tool->snake_name);
  rc = ND_OK;
out:
  free(mv);
  free(pv);
  free(props);
  return rc;
}

static int is_ascii_ws(unsigned char c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == 0x0C;
}

/* top_level_elements: the source span of each element of the top-level array. Returns the count
 * found (at most `max`), or -1 for None or more than `max` elements (both "malformed"). */
static long top_level_elements(const char *text, size_t len, size_t *st, size_t *ln, size_t max) {
  const unsigned char *b = (const unsigned char *)text;
  size_t open = 0, i, depth = 0, start = 0, count = 0;
  int in_str = 0, esc = 0, have_start = 0;
  while (open < len && is_ascii_ws(b[open])) open++;
  if (open == len || b[open] != '[') return -1;
  for (i = open + 1; i < len; i++) {
    unsigned char c = b[i];
    if (in_str) {
      if (esc)
        esc = 0;
      else if (c == '\\')
        esc = 1;
      else if (c == '"')
        in_str = 0;
      continue;
    }
    if (c == '"') {
      in_str = 1;
      if (!have_start) {
        have_start = 1;
        start = i;
      }
    } else if (c == '{' || c == '[') {
      if (!have_start) {
        have_start = 1;
        start = i;
      }
      depth++;
    } else if ((c == '}' || c == ']') && depth > 0) {
      depth--;
    } else if ((c == ',' || c == ']') && depth == 0) {
      if (have_start) {
        size_t end;
        have_start = 0;
        if (count == max) return -1;
        nd_rust_trim_end(text + start, i - start, &end);
        st[count] = start;
        ln[count] = end;
        count++;
      }
      if (c == ']') return (long)count;
    } else if (!is_ascii_ws(c)) {
      if (!have_start) {
        have_start = 1;
        start = i;
      }
    }
  }
  return -1;
}

static int finish_err(nd_json_writer *e, int rc, char **errmsg) {
  if (rc == ND_E_FORMAT && !e->err && errmsg) {
    *errmsg = e->data ? e->data : dup_cstr("");
    if (!*errmsg) rc = ND_E_NOMEM;
    return rc;
  }
  nd_json_writer_free(e);
  return e->err && rc == ND_E_FORMAT ? ND_E_NOMEM : rc;
}

int nd_catalogue_parse(const char *text, size_t len, nd_catalogue **out, char **errmsg) {
  nd_json_writer e;
  nd_json_doc *doc = NULL;
  const nd_json_value *root;
  nd_catalogue *c = NULL;
  size_t st[ND_MAX_TOOLS], ln[ND_MAX_TOOLS], k, n;
  char *serr = NULL;
  long ns;
  int rc;
  if (errmsg) *errmsg = NULL;
  if (!out || (!text && len)) return ND_E_ARG;
  *out = NULL;
  if (!text) text = "";
  nd_json_writer_init(&e, 0);
  if (!nd_json_utf8_valid((const unsigned char *)text, len)) {
    rc = fail(&e, "catalogue is not valid UTF-8");
    rc = finish_err(&e, rc, errmsg);
    return rc == ND_E_FORMAT ? ND_E_ARG : rc;
  }
  if (len > ND_MAX_CATALOGUE_BYTES) {
    w_fmt(&e, "catalogue is %llu bytes (limit %u)", ZU(len), ND_MAX_CATALOGUE_BYTES);
    return finish_err(&e, e.err ? ND_E_NOMEM : ND_E_FORMAT, errmsg);
  }
  rc = nd_serde_parse(text, len, &doc, &serr);
  if (rc != ND_OK) {
    if (rc == ND_E_FORMAT) {
      w_str(&e, "catalogue: ");
      w_str(&e, serr ? serr : "");
      rc = e.err ? ND_E_NOMEM : ND_E_FORMAT;
    }
    free(serr);
    return finish_err(&e, rc, errmsg);
  }
  root = nd_json_root(doc);
  if (root->type != ND_JSON_ARRAY) {
    rc = fail(&e, "catalogue must be a JSON array of tools");
    goto out;
  }
  n = root->u.arr.len;
  if (n == 0 || n > ND_MAX_TOOLS) {
    w_fmt(&e, "catalogue must hold 1..=%u tools", ND_MAX_TOOLS);
    rc = e.err ? ND_E_NOMEM : ND_E_FORMAT;
    goto out;
  }
  ns = top_level_elements(text, len, st, ln, n);
  if (ns < 0 || (size_t)ns != n) {
    rc = fail(&e, "catalogue array is malformed");
    goto out;
  }
  c = (nd_catalogue *)calloc(1, sizeof *c);
  if (!c || !(c->tools = (nd_tool *)nd_calloc(n, sizeof *c->tools))) {
    rc = ND_E_NOMEM;
    goto out;
  }
  for (k = 0; k < n; k++) {
    nd_tool *t = &c->tools[k];
    size_t j;
    c->n_tools = k + 1;
    rc = parse_tool(&root->u.arr.items[k], text + st[k], ln[k], t, &e);
    if (rc) {
      if (rc == ND_E_FORMAT) {
        /* Prefix "tool {i}: " to what parse_tool wrote. */
        nd_json_writer m;
        nd_json_writer_init(&m, 0);
        w_fmt(&m, "tool %llu: ", ZU(k));
        nd_json_writer_raw(&m, e.data ? e.data : "", e.len);
        nd_json_writer_free(&e);
        e = m;
        if (e.err) rc = ND_E_NOMEM;
      }
      goto out;
    }
    for (j = 0; j < k; j++) {
      const nd_tool *o = &c->tools[j];
      if (bytes_eq(o->name, o->name_len, t->name, t->name_len) ||
          bytes_eq(o->snake_name, o->snake_len, t->snake_name, t->snake_len)) {
        w_fmt(&e, "tool %llu: duplicate name ", ZU(k));
        nd_rust_debug_str(&e, t->name, t->name_len);
        rc = e.err ? ND_E_NOMEM : ND_E_FORMAT;
        goto out;
      }
    }
  }
  rc = ND_OK;
out:
  nd_json_doc_free(doc);
  if (rc == ND_OK) {
    nd_json_writer_free(&e);
    *out = c;
    return ND_OK;
  }
  nd_catalogue_free(c);
  return finish_err(&e, rc, errmsg);
}

const nd_tool *nd_catalogue_get(const nd_catalogue *c, const char *name, size_t len) {
  size_t k;
  if (!c) return NULL;
  for (k = 0; k < c->n_tools; k++)
    if (bytes_eq(c->tools[k].name, c->tools[k].name_len, name, len)) return &c->tools[k];
  return NULL;
}

const nd_tool *nd_catalogue_resolve(const nd_catalogue *c, const char *emitted, size_t len) {
  size_t k;
  const nd_tool *t = nd_catalogue_get(c, emitted, len);
  if (t || !c) return t;
  for (k = 0; k < c->n_tools; k++)
    if (bytes_eq(c->tools[k].snake_name, c->tools[k].snake_len, emitted, len))
      return &c->tools[k];
  return NULL;
}

int nd_catalogue_tools_json(const nd_catalogue *c, const nd_strv *names, size_t n_names,
                            char **out, size_t *out_len, char **errmsg) {
  nd_json_writer w;
  size_t k, j, parts = 0;
  if (errmsg) *errmsg = NULL;
  if (!c || !out || (!names && n_names)) return ND_E_ARG;
  *out = NULL;
  nd_json_writer_init(&w, 0);
  if (names) {
    for (k = 0; k < n_names; k++) {
      if (!nd_catalogue_get(c, names[k].ptr, names[k].len)) {
        w_str(&w, "tool ");
        nd_rust_debug_str(&w, names[k].ptr, names[k].len);
        w_str(&w, " is not in the catalogue");
        return finish_err(&w, w.err ? ND_E_NOMEM : ND_E_FORMAT, errmsg);
      }
    }
  }
  nd_json_writer_raw(&w, "[", 1);
  for (k = 0; k < c->n_tools; k++) {
    const nd_tool *t = &c->tools[k];
    int take = !names;
    for (j = 0; !take && j < n_names; j++)
      take = bytes_eq(names[j].ptr, names[j].len, t->name, t->name_len);
    if (!take) continue;
    if (parts++) nd_json_writer_raw(&w, ",", 1);
    nd_json_writer_raw(&w, t->json, t->json_len);
  }
  nd_json_writer_raw(&w, "]", 1);
  if (w.err) {
    nd_json_writer_free(&w);
    return ND_E_NOMEM;
  }
  *out = w.data;
  if (out_len) *out_len = w.len;
  return ND_OK;
}
