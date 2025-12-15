import { AreaChartData } from "../AreaChart";
import { BarChartData } from "../BarChart";
import { LineChartData } from "../LineChart";
import { ScatterPoint } from "../ScatterChart/types";
export declare const useYAxisLabelWidth: (data: AreaChartData | LineChartData | BarChartData | ScatterPoint[], dataKeys: string[]) => {
    yAxisWidth: number;
    setLabelWidth: (displayValue: string) => void;
};
//# sourceMappingURL=useYAxisLabelWidth.d.ts.map