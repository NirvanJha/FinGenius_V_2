import { ChartColorPalette } from "../../ThemeProvider";
export type ColorPalette = {
    name: string;
    colors: string[];
};
export type PaletteName = "ocean" | "orchid" | "emerald" | "spectrum" | "sunset" | "vivid";
type PaletteMap = Record<string, ColorPalette>;
declare const colorPalettes: PaletteMap;
export type PaletteKey = keyof typeof colorPalettes;
export declare const getPalette: (key: PaletteKey) => ColorPalette;
export declare const getAllPalettes: () => ColorPalette[];
export declare const getPaletteKeys: () => PaletteKey[];
export declare const getPaletteKey: (palette: ColorPalette) => PaletteKey;
export declare const getPaletteMap: () => PaletteMap;
export declare const getDistributedColors: (colors: string[], dataLength: number) => string[];
export declare const useChartPalette: ({ chartThemeName, customPalette, themePaletteName, dataLength, }: {
    chartThemeName: PaletteName;
    customPalette?: string[];
    themePaletteName: keyof ChartColorPalette;
    dataLength: number;
}) => string[];
export {};
//# sourceMappingURL=PalletUtils.d.ts.map