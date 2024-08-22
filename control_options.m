function ctrlOptions = control_options()
    ctrlOptions = struct();
    ctrlOptions.fMax = [10;0];
    ctrlOptions.fSpan = [0,1];
    ctrlOptions.fType = "constant";
    ctrlOptions.alpha = 0.5;
    ctrlOptions.friction = "Specker"; % coulomb friction
    ctrlOptions.tSample = 0.01; % if use coulomb friction, use a fix sample time interval
end
