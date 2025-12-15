import * as Radio from "@radix-ui/react-radio-group";
import { CSSProperties, ReactElement } from "react";
import { RadioItemProps } from "../RadioItem";
type RadioGroupVariant = "clear" | "card" | "sunk";
interface RadioGroupProps extends Radio.RadioGroupProps {
    children: ReactElement<RadioItemProps> | ReactElement<RadioItemProps>[];
    variant?: RadioGroupVariant;
    className?: string;
    style?: CSSProperties;
}
declare const RadioGroup: import("react").ForwardRefExoticComponent<RadioGroupProps & import("react").RefAttributes<HTMLDivElement>>;
export { RadioGroup, type RadioGroupProps };
//# sourceMappingURL=RadioGroup.d.ts.map