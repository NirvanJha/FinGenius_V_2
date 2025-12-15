import { CSSProperties, ReactElement } from "react";
import { SwitchItemProps } from "../SwitchItem";
type SwitchGroupVariant = "clear" | "card" | "sunk";
export interface SwitchGroupProps {
    children: ReactElement<SwitchItemProps> | ReactElement<SwitchItemProps>[];
    className?: string;
    style?: CSSProperties;
    variant?: SwitchGroupVariant;
}
declare const SwitchGroup: import("react").ForwardRefExoticComponent<SwitchGroupProps & import("react").RefAttributes<HTMLDivElement>>;
export { SwitchGroup };
//# sourceMappingURL=SwitchGroup.d.ts.map